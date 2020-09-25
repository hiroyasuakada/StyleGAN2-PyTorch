import numpy as np
import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F

from network_base_layers import Blur, Amplify, AddChannelwiseBias, EqualizedFullyConnect, \
                                PixelwiseNoise, PixelwiseRondomizedNoise, \
                                FusedBlur3x3, EqualizedModConv2D, EqualizedConv2D

from op import FusedLeakyReLU, fused_leaky_relu, upfirdn2d


###################################################################################################
###                          Sub Blocks: ModConvLayer, ToRGB, ConvLayer                         ###
###################################################################################################


class ModConvLayer(nn.Module):
    def __init__(self, dlatent_size, resolution, in_channels, out_channels, padding, stride,
                 kernel_size, up=False, use_noise=True):
        
        super().__init__()

        self.conv = EqualizedModConv2D(dlatent_size=dlatent_size, 
                                       in_channels=in_channels, out_channels=out_channels,
                                       padding=padding, stride=stride,
                                       kernel_size=kernel_size, up=up)  # no need of "use_noise" for now
        self.noise = PixelwiseNoise(resolution=resolution)
        self.bias = AddChannelwiseBias(out_channels=out_channels, lr=1.0)
        self.amplify = Amplify(rate=2**0.5)
        self.activate = nn.LeakyReLU(negative_slope=0.2)

    def forward(self, x, style):
        x = self.conv(x, style)
        x = self.noise(x)
        x = self.bias(x)
        x = self.amplify(x)
        x = self.activate(x)

        return x


class ToRGB(nn.Module):
    def __init__(self, dlatent_size, in_channels, num_channels, resample_kernel=None):
        super().__init__()

        # if resample_kernel is None:
        #     resample_kernel =  [1, 3, 3, 1]
        
        # self.upsample = Upsample()
        self.conv = EqualizedModConv2D(dlatent_size=dlatent_size, 
                                       in_channels=in_channels, out_channels=num_channels,
                                       kernel_size=1, padding=0, stride=1,
                                       demodulate=False)
        self.bias = AddChannelwiseBias(out_channels=num_channels, lr=1.0)

    def forward(self,x, style, skip):
        x = self.conv(x, style)
        out = self.bias(x)
        
        if skip is not None:  # architecture = 'skip'
            out = out + F.interpolate(skip, scale_factor=2, mode='bilinear',align_corners=False)

        return out


class ConvLayer(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, downsample=False, 
                 blur_kernel=[1, 3, 3, 1], bias=True, activate=True):

        super().__init__()

        if downsample:
            factor = 2
            p = (len(blur_kernel) - factor) + (kernel_size - 1)
            pad0 = (p + 1) // 2
            pad1 = p // 2

            self.blur = Blur(blur_kernel, pad=(pad0, pad1))

            stride = 2
            self.padding = 0

        else:
            stride = 1
            self.padding = kernel_size // 2

        self.conv = EquilizedConv2D(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size,
                                    padding=padding, stride=stride, bias=False)

        if activate:
            self.bias = AddChannelwiseBias(out_channels=out_channels, lr=1.0)
            self.amplify = Amplify(rate=2**0.5)
            self.activate = nn.LeakyReLU(negative_slope=0.2)

    def forward(self, x):
        if downsample:
            x = self.blur(x)

        x = self.conv(x)

        if activate:
            x = self.bias(x)
            x = self.amplify(x)
            x = self.activate(x)

        return x


###################################################################################################
###                        Main Blocks: Mapping, Input, Synthesis, Resnet                       ###
###################################################################################################


class GeneratorMappingBlock(nn.Module):
    """
    Build blocks for main mapping layers.
    """

    def __init__(self, in_fmaps, out_fmaps):
        super().__init__()

        self.fc = EqualizedFullyConnect(in_dim=in_fmaps, out_dim=out_fmaps, lr=0.01)
        self.bias = AddChannelwiseBias(out_channels=out_fmaps, lr=0.01)
        self.amplify = Amplify(rate=2**0.5)
        self.activate = nn.LeakyReLU(negative_slope=0.2)
    
    def forward(self, x):
        x = self.fc(x)
        x = self.bias(x)
        x = self.amplify(x)
        x = self.activate(x)

        return x
        

class InputBlock(nn.Module):
    def __init__(self, dlatent_size, num_channels, in_fmaps, out_fmaps, use_noise):
        super().__init__()

        # generate input
        self.const_input = nn.Parameter(torch.randn(1, in_fmaps, 4, 4), requires_grad=True)

        # build layers for first input
        self.conv = ModConvLayer(dlatent_size=dlatent_size, resolution=4,
                                 in_channels=in_fmaps, out_channels=out_fmaps,
                                 kernel_size=3, padding=1, stride=1,
                                 use_noise=use_noise)
        self.to_rgb = ToRGB(dlatent_size=dlatent_size,
                            in_channels=out_fmaps,
                            num_channels=num_channels)

    def forward(self, dlatents_in):
        """
        Args:
            dlatents_in: Input: Disentangled latents (W) [minibatch, num_layers, dlatent_size]. ex. [4, 18, 512]
        """
        x = self.const_input.repeat(dlatents_in.shape[0], 1, 1, 1)
        x = self.conv(x, dlatents_in[:, 0])
        skip = self.to_rgb(x, dlatents_in[:, 1], None)

        return x, skip


class GeneratorSynthesisBlock(nn.Module):
    """
    Build blocks for main synthesis layers.
    """

    def __init__(self, dlatent_size, num_channels, res, 
                 in_fmaps, out_fmaps, use_noise):
        super().__init__()

        self.res = res
        self.conv0_up = ModConvLayer(dlatent_size=dlatent_size, resolution=2 ** self.res,
                                     in_channels=in_fmaps, out_channels=out_fmaps,
                                     kernel_size=3, padding=0, stride=2,
                                     up=True, use_noise=use_noise)
        self.conv1 = ModConvLayer(dlatent_size=dlatent_size, resolution=2 ** self.res,
                                  in_channels=out_fmaps, out_channels=out_fmaps,
                                  kernel_size=3, padding=1, stride=1,
                                  use_noise=use_noise)
        self.to_rgb = ToRGB(dlatent_size=dlatent_size,
                            in_channels=out_fmaps, num_channels=num_channels)
    
    def forward(self, x, dlatents_in, skip):
        x = self.conv0_up(x, dlatents_in[:, self.res * 2 - 5])
        x = self.conv1(x, dlatents_in[:, self.res * 2 - 4])

        # architecture='skip'
        skip = self.to_rgb(x, dlatents_in[:, self.res * 2 - 3], skip)

        return x, skip


class DiscriminatorBlock(nn.Module):
    """
    Build blocks for discriminator (resnets).
    """
    
    def __init__(self, in_fmaps, out_fmaps):
        super().__init__()

        self.conv0 = ConvLayer(in_channels=in_fmaps, out_channels=in_fmaps, kernel_size=3)
        self.conv1_down = ConvLayer(in_channels=in_fmaps, out_channels=out_fmaps, kernel_size=3, downsample=True)

        self.skip = ConvLayer(in_channels=in_fmaps, out_channels=out_fmaps, downsample=True, activate=False, bias=False)

    def forward(self, x):
        out = self.conv0(x)
        out = self.conv1_down(out)

        skip = self.skip(x)
        out = (out + skip) / math.sqrt(2)

        return out   
        