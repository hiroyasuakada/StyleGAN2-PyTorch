import numpy as np
import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F


class PixelwiseNormalization(nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x):
        return x / torch.sqrt((x**2).mean(1, keepdim=True) + 1e-8)


class TruncationTrick(nn.Module):
    def __init__(self, num_target, threshold, output_num, style_dim):
        super().__init__()
        self.num_target = num_target
        self.threshold = threshold
        self.output_num = output_num
        self.style_dim = style_dim
        self.register_buffer('avg_style', torch.zeros((style_dim, )))

    def forward(self, x):
        N, D = x.shape
        O = self.output_num
        x = x.view(N, 1, D).expand(N, O, D)
        rate = torch.cat([torch.ones(N, self.num_target,   D) * self.threshold, 
                          torch.ones(N, O-self.num_target, D) * 1.0])
        avg = self.avg_style.view(1, 1, D).expand(N, O, D)
        return avg + (x - avg) * rate


class Amplify(nn.Module):
    def __init__(self, rate):
        supre().__init__()
        self.rate = rate

    def forward(self, x):
        return x * self.rate


class AddChannelwiseBias(nn.Module):
    def __init__(self,out_channels, lr):
        super().__init__()
        # lr = 1.0 (conv,mod,AdaIN), 0.01 (mapping)
        self.bias = nn.Parameter(torch.zeros(out_channels))
        torch.nn.init.zeros_(self.bias.data)
        self.bias_scalar = lr

    def forward(self, x):
        oC, _ = self.bias.shape
        shape = (1, oC) if x.ndim==2 else (1, oC, 1, 1)
        y = x + self.bias.view(*shape) * self.bias_scalar
        return y


class PixelwiseNoise(nn.Module):
    def __init__(self, resolution):
        super().__init__()
        self.register_buffer('const_noise', torch.randn((1, 1, resolution, resolution, )))
        self.noise_scaler = nn.Parameter(torch.zeros(1))

    def forward(self, x):
        N, C, H, W = x.shape
        noise = self.const_noise.expand(N, C, H, W)
        return x + noise * self.noise_scaler


class DenseLayer(nn.Module):
    def __init__(self, in_dim, out_dim, lr):
        super().__init__()
        # lr = 0.01 (mapping), 1.0 (mod,AdaIN)
        self.weight = nn.Parameter(torch.randn((out_dim, in_dim)))
        torch.nn.init.normal_(self.weight.data, mean=0.0, std=1.0/lr)
        self.weight_scaler = 1/(in_dim ** 0.5) * lr

    def forward(self, x):
        # x (N,D)
        return F.linear(x, self.weight * self.weight_scaler, None)


class 



class GeneratorMapping(nn.Module):
    def __init__(self, latent_size=512, dlatent_size=512, mapping_layers=8, mapping_dim=512, 
                 mapping_lr=0.01, mapping_nonlinearity='lrelu', nomalize_latents=True, **_kwargs):

        """
        Mapping network used in the StyleGAN paper.
        :param latent_size: Latent vector(Z) dimensionality.
        :param dlatent_size: Disentangled latent (W) dimensionality.
        :param mapping_layers: Number of mapping layers.
        :param mapping_dim: Number of dimensionality in the mapping layers.
        :param mapping_lr: Learning rate for the mapping layers.
        :param mapping_nonlinearity: Activation function: 'relu', 'lrelu'.
        :param normalize_latents: Normalize latent vectors (Z) before feeding them to the mapping layers?
        :param _kwargs: Ignore unrecognized keyword args.
        """

        super().__init__()
        self.latent_size = latent_size
        self.dlatent_size = dlatent_size
        self.mapping_dim = mapping_dim
        self.normalize_latents = normalize_latents

        intermediate_layers = []
        for layer_idx in range(0, mapping_layers):
            input_size = self.latent_size if layer_idx==0 else self.mapping_dim
            output_size = self.dlatent_size if layer_idx==0 else self.mapping_dim

            layers.append('fc', 'bias', 'amp', 'lrelu')
        
        self.model = nn.Sequential(layers)




class GeneratorSynthesisStylegan2(nn.Module):
    def __init__(self):
        pass

print()