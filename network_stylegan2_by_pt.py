import numpy as np
import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F

from network_base_by_pt import PixelwiseNormalization, TruncationTrick
from network_base_by_pt import EqualizedFullyConnect, AddChannelwiseBias, Amplify
from network_base_by_pt import GeneratorSynthesisBlock, InputBlock


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
            input_fmaps = self.latent_size if layer_idx==0 else self.mapping_fmaps
            output_fmaps = self.dlatent_size if layer_idx==0 else self.mapping_fmaps

            layers.append(EqualizedFullyConnect(in_dim=input_fmaps, out_dim=output_fmaps, lr=0.01))
            layers.append(AddChannelwiseBias(out_channels=output_fmaps, lr=0.01))
            layers.append(Amplify(rate=2**0.5))
            layers.append(nn.LeakyReLU(negative_slope=0.2))

        # build a last layer for truncation trick
        layers.append(TruncationTrick(num_target=10, threshold=0.7, output_num=18, style_dim=512))
        
        self.model = nn.ModuleList(layers)

        # # display layers
        # print('Generator Mapping Network: ')
        # print(self.model)

    def forward(self, latent_in):
        for i in range(len(self.model)):
            x = self.model[i](x)
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

        # early layers
        pass



class Generator(nn.Module):
    def __init__(self, resolution=1024, latent_size=512, dlatent_size=512, **_kwargs):
        
        """
        Args:
            latent_size: Latent vector(Z) dimensionality.
            dlatent_size: Disentangled latent (W) dimensionality.
        """

        super().__init__()

        # synthetic rate for synthetic images / do not use this time
        self.register_buffer('style_mixing_rate', torch.zeros((1,)))

        # set up sub networks
        self.mapping_network = GeneratorMapping(latent_size=latent_size, dlatent_size=dlatent_size)
        self.synthesis_network = GeneratorSynthesisStylegan2()

    def forward(self,latents_in, return_dlatents=False):

        """
        Args:
            latent_size: Latent vector(Z) dimensionality.
            dlatent_size: Disentangled latent (W) dimensionality.

        returns: images (and dlatents)
        """

        dlatents_in = self.mapping_network(latents_in)
        images_out = self.synthesis_network(dlatents_in)

        return images_out if return_dlatents else images_out, dlatents_in


class Discriminator(nn.Module):
    def __init__(self, num_channels=3, resolution=1024):
        pass


if __name__ == '__main__':
    x = GeneratorMapping()
