import numpy as np
import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F

import time
from tensorboardX import SummaryWriter

from model_base_layers import PixelwiseNormalization, TruncationTrick
from model_blocks import GeneratorMappingBlock, GeneratorSynthesisInputBlock, GeneratorSynthesisBlock, \
                         DiscriminatorBlock, DiscriminatorLastBlock, ConvLayer
# from model_loss import                        
# from dataset import Dataset


###################################################################################################
###                       Sub Networks: Generator Mapping and Synthesis                         ###
###################################################################################################


class GeneratorMapping(nn.Module):
    def __init__(self, latent_size=512, dlatent_size=512, mapping_layers=8, mapping_fmaps=512, 
                 mapping_lr=0.01, mapping_nonlinearity='lrelu', normalize_latents=True, **_kwargs):

        """
        Mapping network used in the StyleGAN paper.
            latent_size: Latent vector(Z) dimensionality.
            dlatent_size: Disentangled latent (W) dimensionality.
            mapping_layers: Number of mapping layers.
            mapping_fmaps: Number of activations in the mapping layers.
            mapping_lr: Learning rate for the mapping layers.
            mapping_nonlinearity: Activation function: 'relu', 'lrelu'.
            normalize_latents: Normalize latent vectors (Z) before feeding them to the mapping layers?
            **_kwargs: Ignore unrecognized keyword args.
        """

        super().__init__()

        self.latent_size = latent_size
        self.dlatent_size = dlatent_size
        self.mapping_fmaps = mapping_fmaps
        self.normalize_latents = normalize_latents

        layers = []

        # build a first layer for pixel wise normalization
        layers.append(PixelwiseNormalization())

        # build intermediate layers
        for layer_idx in range(0, mapping_layers):
            in_fmaps = self.latent_size if layer_idx==0 else self.mapping_fmaps
            out_fmaps = self.dlatent_size if layer_idx==0 else self.mapping_fmaps

            layers.append(GeneratorMappingBlock(in_fmaps=in_fmaps, out_fmaps=out_fmaps))

        # build a last layer for truncation trick
        layers.append(TruncationTrick(num_target=10, threshold=0.7, out_num=18, dlatent_size=512))
        
        self.blocks = nn.ModuleList(layers)

        # # display layers
        # print('Generator Mapping Network: ')
        # print(self.model)

    def forward(self, latents_in):

        """
        Args:
            latents_in: First input: Latent vectors (Z) [minibatch, latent_size]. ex, [4, 512]
        """

        x = latents_in
        for i in range(len(self.blocks)):
            x = self.blocks[i](x)
        return x


class GeneratorSynthesis(nn.Module):
    def __init__(self, dlatent_size=512, num_channels=3, resolution=1024,
                 fmap_base=16 << 10, fmap_decay=1.0, fmap_min=1, fmap_max=512,
                 randomize_noise=True, architecture='skip', **_kwargs):
        
        """
        Args:
            dlatent_size: Disentangled latent (W) dimensionality.
            num_channels: Number of output color channels.
            resolution: Output resolution.
            fmap_base: Overall multiplier for the number of feature maps.
            fmap_decay: log2 feature map reduction when doubling the resolution.
            fmap_min: Minimum number of feature maps in any layer.
            fmap_max: Maximum number of feature maps in any layer.
            randomize_noise: True = randomize noise inputs every time (non-deterministic),
                             False = read noise inputs from variables.
            architecture: 'orig', 'skip', 'resnet'.
            **_kwargs: Ignore unrecognized keyword args.):
        """

        super().__init__()

        resolution_log2 = int(np.log2(resolution))
        assert resolution == 2 ** resolution_log2 and resolution >= 4
        assert architecture in ['orig', 'skip', 'resnet']

        self.architecture = architecture
        self.resolution_log2 = resolution_log2
        self.randomize_noise = randomize_noise

        def nf(stage):
            return np.clip(int(fmap_base / (2.0 ** (stage * fmap_decay))), fmap_min, fmap_max)

        # build early layers
        # nf(1) = 512
        self.init_block = GeneratorSynthesisInputBlock(dlatent_size=dlatent_size, num_channels=num_channels,
                                                       in_fmaps=nf(1), out_fmaps=nf(1), use_noise=randomize_noise)

        # build all the remaining layers
        # blocks / in / out
        #    0     512  512
        #    1     512  512
        #    2     512  512
        #    3     512  512
        #    4     512  256
        #    5     256  128
        #    6     128   64
        #    7      64   32
        blocks = [GeneratorSynthesisBlock(dlatent_size=dlatent_size, num_channels=num_channels, res=res,
                                  in_fmaps=nf(res - 2), out_fmaps=nf(res - 1), use_noise=randomize_noise)
                  for res in range(3, resolution_log2 + 1)]

        self.blocks = nn.ModuleList(blocks)

    def forward(self, dlatents_in):
        """
        Args:
            dlatents_in: Input: Disentangled latents (W) [minibatch, num_layers, dlatent_size]. ex. [4, 18, 512]

        Returns:
            images
        """

        x, skip = self.init_block(dlatents_in)

        for block in self.blocks:
            x, skip = block(x, dlatents_in, skip)

        images = skip

        return images


###################################################################################################
###                          Main Networks: Generator and Discriminator                         ###
###################################################################################################


class Generator(nn.Module):
    def __init__(self, resolution=1024, latent_size=512, dlatent_size=512, **_kwargs):
        
        """
        Args:
            latent_size: Latent vector(Z) dimensionality.
            dlatent_size: Disentangled latent (W) dimensionality.
        """

        super().__init__()

        ## synthetic rate for synthetic images / do not use this time
        # self.register_buffer('style_mixing_rate', torch.zeros((1,)))

        # set up sub networks
        self.mapping_network = GeneratorMapping(latent_size=latent_size, dlatent_size=dlatent_size, **_kwargs)
        self.synthesis_network = GeneratorSynthesis(resolution=resolution, **_kwargs)

    def forward(self,latents_in, return_dlatents=False):

        """
        Args:
            latents_in: First input: Latent vectors (Z) [minibatch, latent_size].
            return_dlatents: Return dlatents in addition to the images?
            (labels_in: Second input: Conditioning labels [minibatch, label_size].)
        Returns: images
        """

        dlatents_in = self.mapping_network(latents_in)
        imgs_out = self.synthesis_network(dlatents_in)

        if return_dlatents:
            return imgs_out, dlatents_in
        else:
            return imgs_out


class Discriminator(nn.Module):
    def __init__(self, num_channels=3, resolution=1024, 
                 fmap_base=16 << 10, fmap_decay=1.0, fmap_min=1, fmap_max=512, 
                 architecture='resnet', nonlinearity='lrelu', 
                 mbstd_group_size=4, mbstd_num_features=1, resample_kernel=None, **_kwargs):

        """
        Args:
            num_channels: Number of input color channels. Overridden based on dataset.
            resolution: Input resolution. Overridden based on dataset.
            label_size: Dimensionality of the labels, 0 if no labels. Overridden based on dataset.
            fmap_base: Overall multiplier for the number of feature maps.
            fmap_decay: log2 feature map reduction when doubling the resolution.
            fmap_min: Minimum number of feature maps in any layer.
            fmap_max: Maximum number of feature maps in any layer.
            architecture: Architecture: 'orig', 'skip', 'resnet'.
            nonlinearity: Activation function: 'relu', 'lrelu', etc.
            mbstd_group_size: Group size for the minibatch standard deviation layer, 0 = disable.
            mbstd_num_features: Number of features for the minibatch standard deviation layer.
            resample_kernel: Low-pass filter to apply when resampling activations. None = no filtering.
            **_kwargs: Ignore unrecognized keyword args.):
        """

        super().__init__()

        resolution_log2 = int(np.log2(resolution))
        assert resolution == 2 ** resolution_log2 and resolution >= 4
        assert architecture in ['orig', 'skip', 'resnet']  # we use skip in this implementation

        self.architecture = architecture
        self.resolution_log2 = resolution_log2
        self.nonlinearity = nonlinearity

        def nf(stage):
            return np.clip(int(fmap_base / (2.0 ** (stage * fmap_decay))), fmap_min, fmap_max)
            # blocks / in / out
            #    0      32   64
            #    1      64  128
            #    2     128  256
            #    3     256  512
            #    4     512  512
            #    5     512  512
            #    6     512  512
            #    7     512  512

        # build a first layer for images [N, 3, 1024, 1024]
        self.init_conv = ConvLayer(in_channels=3, out_channels=nf(resolution_log2 - 1), kernel_size=1)
            
        # build main layers
        blocks = [DiscriminatorBlock(in_fmaps=nf(res - 1), out_fmaps=nf(res - 2)) 
                  for res in range(resolution_log2, 2, -1)]

        self.blocks = nn.ModuleList(blocks)

        # build a last layer
        self.last_block = DiscriminatorLastBlock(in_fmaps=nf(1), in_fmaps_fc=nf(0), 
                                                 mbstd_group_size=mbstd_group_size, 
                                                 mbstd_num_features=mbstd_num_features)                                  

    def forward(self, imgs_in):
        out = self.init_conv(imgs_in)

        for block in self.blocks:
            out = block(out)

        out = self.last_block(out)

        return out


if __name__ == '__main__':

    # ## mapping_network
    # g_mapping = GeneratorMapping().to('cuda:0')
    # print(g_mapping)
    # test_latents_in = torch.randn(4, 512).to('cuda:0')
    # test_dlatents_out = g_mapping(test_latents_in)
    # print('dlatents_out: {}'.format(test_dlatents_out.shape))  # [N 18, D] = [4, 18, 512]
    # # check params
    # params_0 = 0
    # for p in g_mapping.parameters():
    #     if p.requires_grad:
    #         params_0 += p.numel()
    # print('params_0: {}'.format(params_0))


    # # synthesis_network
    # g_synthesis = GeneratorSynthesis().to('cuda:0')
    # print(g_synthesis)
    # test_dlatents_in = torch.randn(4, 18, 512).to('cuda:0')
    # test_imgs_out = g_synthesis(test_dlatents_in)
    # print('imgs_out: {}'.format(test_imgs_out.shape))  # [4, 3, 1024, 1024]
    # # check params
    # params_1 = 0
    # for p in g_synthesis.parameters():
    #     if p.requires_grad:
    #         params_1 += p.numel()
    # print('params_0: {}'.format(params_1))


    # # generator_network
    # Gen = Generator().to('cuda:0')
    # print(Gen)
    # test_latents_in = torch.randn(4, 512).to('cuda:0')
    # test_imgs_out = Gen(test_latents_in)
    # print('imgs_out: {}'.format(test_imgs_out.shape))  # [4, 3, 1024, 1024]
    # # check params
    # params_Gen = 0
    # for p in Gen.parameters():
    #     if p.requires_grad:
    #         params_Gen += p.numel()
    # print('params_Gen: {}'.format(params_Gen))


    # discriminator_network
    Dis = Discriminator().to('cuda:0')
    print(Dis)
    test_imgs_in = torch.randn(4, 3, 1024, 1024).to('cuda:0')
    print(test_imgs_in.shape)
    out = Dis(test_imgs_in)
    print('out: {}'.format(out.shape))
    # check params
    params_Dis = 0
    for p in Dis.parameters():
        if p.requires_grad:
            params_Dis += p.numel()
    print('params_Dis: {}'.format(params_Dis))



    # # check fmaps
    # fmap_base=16 << 10
    # fmap_decay=1.0
    # fmap_min=1
    # fmap_max=512
    # resolution = 1024
    # resolution_log2 = int(np.log2(resolution))

    # def nf(stage):
    #     return np.clip(int(fmap_base / (2.0 ** (stage * fmap_decay))), fmap_min, fmap_max)

    # for res in range(resolution_log2, 2, -1):
    #     print(res)
    #     print("in_fmap: {}".format(nf(res - 1)))
    #     print("out_fmap: {}".format(nf(res - 2)))
    #     print('=============================')

    # print(nf(0))