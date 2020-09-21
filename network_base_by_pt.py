import numpy as np
import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F


class PixelwiseNormalization(nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x):
        x = x / torch.sqrt((x**2).mean(1, keepdim=True) + 1e-8)
        return x


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
        avg += (x - avg) * rate
        return avg


class Amplify(nn.Module):
    def __init__(self, rate):
        super().__init__()
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


class EqualizedFullyConnect(nn.Module):
    def __init__(self, in_dim, out_dim, lr):
        super().__init__()
        # lr = 0.01 (mapping), 1.0 (mod,AdaIN)
        self.weight = nn.Parameter(torch.randn((out_dim, in_dim)))
        torch.nn.init.normal_(self.weight.data, mean=0.0, std=1.0/lr)
        self.weight_scaler = 1/(in_dim ** 0.5) * lr

    def forward(self, x):
        # x (N,D)
        x = F.linear(x, self.weight * self.weight_scaler, None)
        return x

## read noise inputs from variables
# class PixelwiseNoise(nn.Module):
#     def __init__(self, resolution):
#         super().__init__()
#         self.register_buffer('const_noise', torch.randn((1, 1, resolution, resolution, )))
#         self.noise_scaler = nn.Parameter(torch.zeros(1))

#     def forward(self, x):
#         N, C, H, W = x.shape
#         noise = self.const_noise.expand(N, C, H, W)
#         x += noise * self.noise_scaler
#         return x

# rondomize noise inputs every time (non-deterministic)
class PixelwiseRondomizedNoise(nn.Module):
    def __init__(self):
        super().__init__()

        self.noise_scaler = nn.Parameter(torch.zeros(1), requires_grad=True)

    def forward(self, x):
        N, C, H, W = x.shape
        noise_inputs = x.new_empty(N, 1, H, W).normal_()
        x += noise_inputs * self.noise_scaler
        return x


# upscaling layer
class FusedBlur3x3(nn.Module):
    def __init__(self):
        super().__init__()
        kernel = np.array([ [1/16, 2/16, 1/16],
                            [2/16, 4/16, 2/16],
                            [1/16, 2/16, 1/16]],dtype=np.float32)
        pads = [[(0,1),(0,1)],[(0,1),(1,0)],[(1,0),(0,1)],[(1,0),(1,0)]]
        kernel = np.stack( [np.pad(kernel,pad,'constant') for pad in pads] ).sum(0)
        #kernel [ [1/16, 3/16, 3/16, 1/16,],
        #         [3/16, 9/16, 9/16, 3/16,],
        #         [3/16, 9/16, 9/16, 3/16,],
        #         [1/16, 3/16, 3/16, 1/16,] ]
        self.kernel = torch.from_numpy(kernel)

    def forward(self, feature):
        # featureは(N,C,H+1,W+1)
        kernel = self.kernel.clone().to(feature.device)
        _N,C,_Hp1,_Wp1 = feature.shape
        x = F.conv2d(feature, kernel.expand(C,1,4,4), padding=1, groups=C)
        return x


class EqualizedMOdConv2D(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, 
                 style_dim, padding, stride, demodulate=True, lr=1):



                 


class ModConvLayer(nn.Module):
    def __init__(self, dlatent_size, input_channels, output_channels, 
                 kernel, up=False, down=False, use_noise=True):
        
        super().__init__()

        self.conv = EqualizedModConv2D(pass)
        self.noise = PixelwiseRondomizedNoise()
        self.bias = AddChannelwiseBias(output_channels=output_channels, lr=1.0)
        self.amplify = Amplify(rate=2**0.5)
        self.activation = nn.LeakyReLU(negative_slope=0.2)

    def forward(self, x, dlatents_in_range):
        x = self.conv(x, dlatents_in_range)
        x = self.noise(x)
        x = self.bias(x)
        x = self.amplify(x)
        x = self.activation(x)
        return x


class InputBlock(nn.Module):
    def __init__(self, dlatent_size, num_channels, input_maps, output_fmaps, use_noise):
        super().__init__()

        # generate input
        self.const_input = nn.Parameter(torch.randn(1, input_fmaps, 4, 4), requires_grad=True)

        # build layers for first input
        self.conv = ModConvLayer(dlatent_size=dlatent_size, 
                                 input_channels=input_fmaps,
                                 output_channels=output_fmaps,
                                 kernel=3, use_noise=use_noise)
        self.to_rgb = ToRGB(dlatent_size=dlatent_size,
                            input_channels=output_fmaps,
                            num_channels=num_channels)

    def forward(self, dlatents_input):
        x = self.const_input.repeat(dlatents_input.shape[0], 1, 1, 1)
        x = self.conv(x, dlatents_input[:, 0])
        y = self.to_rgb(x, dlatents_input[:, 1])
        return x, y


class GeneratorSynthesisBlock(nn.Module):
    """
    Build blocks for main layers.
    """

    def __init__(self, dlatent_size, num_channels, resolution, 
                 input_channels, output_channels, use_noise):
        super().__init__()

        self.resolution = resolution
        self.conv0_up = 
        self.conv1 = ModConvLayer
        self.to_rgb = ToRGB()




if __name__ == '__main__':
    x = GeneratorMapping()
