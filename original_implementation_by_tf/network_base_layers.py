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
    def __init__(self, num_target, threshold, out_num, dlatent_size):
        super().__init__()
        self.num_target = num_target
        self.threshold = threshold
        self.out_num = out_num
        self.dlatent_size = dlatent_size
        self.register_buffer('avg_style', torch.zeros((dlatent_size, )))

    def forward(self, x):
        N, D = x.shape
        O = self.out_num
        x = x.view(N, 1, D).expand(N, O, D)
        rate = torch.cat([torch.ones((N, self.num_target,   D)) * self.threshold, 
                          torch.ones((N, O - self.num_target, D)) * 1.0], 1).to(x.device)
        avg = self.avg_style.view(1, 1, D).expand(N, O, D)
        return avg + (x - avg) * rate


class Amplify(nn.Module):
    def __init__(self, rate):
        super().__init__()
        self.rate = rate

    def forward(self, x):
        x = x * self.rate
        return x


class AddChannelwiseBias(nn.Module):
    def __init__(self,out_channels, lr):
        super().__init__()
        # lr = 1.0 (conv,mod,AdaIN), 0.01 (mapping)
        self.bias = nn.Parameter(torch.zeros(out_channels))
        torch.nn.init.zeros_(self.bias.data)
        self.bias_scalar = lr

    def forward(self, x):
        oC, *_ = self.bias.shape
        shape = (1, oC) if len(x.shape)==2 else (1, oC, 1, 1)  # x.ndim is not supported by Pytorch 1.1, so use len(x.shape)
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

# read noise inputs from variables
class PixelwiseNoise(nn.Module):
    def __init__(self, resolution):
        super().__init__()
        self.register_buffer('const_noise', torch.randn((1, 1, resolution, resolution, )))
        self.noise_scaler = nn.Parameter(torch.zeros(1))

    def forward(self, x):
        N, C, H, W = x.shape
        noise = self.const_noise.expand(N, C, H, W)
        y = x + noise * self.noise_scaler
        return y

# rondomize noise inputs every time (non-deterministic)
class PixelwiseRondomizedNoise(nn.Module):
    def __init__(self):
        super().__init__()

        self.noise_scaler = nn.Parameter(torch.zeros(1), requires_grad=True)

    def forward(self, x):
        N, C, H, W = x.shape
        noise_inputs = x.new_empty(N, 1, H, W).normal_()
        y = x + noise_inputs * self.noise_scaler
        return y


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


class EqualizedModConv2D(nn.Module):
    def __init__(self, dlatent_size, in_channels, out_channels, kernel_size, 
                 padding, stride, up=False, demodulate=True, lr=1):

        super().__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.up = up
        self.padding = padding
        self.stride = stride
        self.demodulate = demodulate

        self.weight = nn.Parameter(torch.randn(out_channels, in_channels, kernel_size, kernel_size))
        torch.nn.init.normal_(self.weight.data, mean=0.0, std=1.0/lr)
        self.weight_scaler = 1 / (in_channels * kernel_size * kernel_size) ** 0.5 * lr

        self.fc = EqualizedFullyConnect(dlatent_size, in_channels, lr)
        self.bias = AddChannelwiseBias(in_channels, lr)
        
        # only for ConvTranspose2D
        self.bulr = FusedBlur3x3()

    def forward(self, x, style):
        N, iC, H, W = x.shape
        oC, iC, kH, kW = self.weight.shape

        # modulate
        mod_rates = self.bias(self.fc(style)) + 1  # (N, iC)
        modulated_weight = self.weight_scaler * self.weight.view(1,oC,iC,kH,kW) * mod_rates.view(N,1,iC,1,1) # (N,oC,iC,kH,kW)
        
        # demodulate
        if self.demodulate:
            # Scaling facotr
            demod_norm = 1 / ((modulated_weight ** 2).sum([2, 3, 4]) + 1e-8) ** 0.5  # (N, oC)
            weight = modulated_weight * demod_norm.view(N, oC, 1, 1, 1)  # (N, oC, iC, kH, kW)
        else:
            # ToRGB
            weight = modulated_weight

        # conv2d or convT2d
        if self.up:
            x = x.view(1, N * iC, H, W)
            weight = weight.view(N * iC, oC, kH, kW)
            out = F.conv_transpose2d(x, weight, padding=self.padding, stride=self.stride, groups=N)
            _, _, H1, W1 = out.shape
            out = out.view(N, oC, H1, W1)
            out = self.bulr(out)
        else:
            x = x.view(1, N * iC, H, W)
            weight = weight.view(N * oC, iC, kH, kW)
            out = F.conv2d(x, weight, padding=self.padding, stride=self.stride, groups=N)
            out = out.view(N, oC, H, W)
        return out
                 
