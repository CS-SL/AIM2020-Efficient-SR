import torch
import torch.nn as nn
import torch.nn.functional as F
import math 
from torch.nn.parameter import Parameter

def get_gaussian_kernel(kernel_size=9, sigma=1.3, channels=3):
    '''get a gaussian blur kernel
       kernel_size = int(math.ceil(1.3 * 3) * 2 + 1), if not kernel_size
    '''
    if kernel_size == None:
        kernel_size = int(math.ceil(sigma * 3) * 2 + 1)
    if sigma == None:
        sigma = 0.3 * ((kernel_size-1)*0.5 - 1) + 0.8
    
    padding = kernel_size // 2 
    x_coord = torch.arange(kernel_size)
    x_grid = x_coord.repeat(kernel_size).view(kernel_size, kernel_size)
    y_grid = x_grid.t()
    xy_grid = torch.stack([x_grid, y_grid], dim=-1).float()

    mean = (kernel_size - 1) / 2.
    variance = sigma ** 2

    # Calculate the 2-dimensional gaussian kernel which is
    # the product of two gaussian distributions for two different
    # variables (in this case called x and y)
    gaussian_kernel = (1. / (math.sqrt(2. * math.pi * variance))) *\
                        torch.exp(- torch.sum((xy_grid - mean)**2., dim=-1) / (2 * variance))

    # Make sure sum of values in gaussian kernel equals 1.
    gaussian_kernel = gaussian_kernel / torch.sum(gaussian_kernel)

    # Reshape for Conv operation
    gaussian_kernel = gaussian_kernel.view(1, 1, kernel_size, kernel_size)
    gaussian_kernel = gaussian_kernel.repeat(channels, 1, 1, 1)

    gaussian_filter = nn.Conv2d(channels, channels, kernel_size=kernel_size, groups=channels, padding=padding, bias=False)
    # gaussian_filter = nn.Conv2d(channels, channels, kernel_size=kernel_size, groups=channels, dilation=7, padding=7, bias=False)

    gaussian_filter.weight.data = gaussian_kernel
    gaussian_filter.weight.requires_grad = False

    return gaussian_filter


class SoftThreshold(nn.Module):
    def __init__(self, theta=0.1):
        super(SoftThreshold, self).__init__()

        self.theta = nn.Parameter(torch.tensor(theta), requires_grad=True)
        self.relu = nn.ReLU()

    def forward(self, x):
        x_ = torch.abs(x) - self.theta
        x = torch.sign(x) * self.relu(x_)
        return x

class SparseBlock(nn.Module):
    def __init__(self, in_feat=3, out_feat=32):
        super(SparseBlock, self).__init__()

        self.g = nn.Conv2d(in_feat, out_feat, kernel_size=3, stride=1, dilation=5, padding=5, bias=False)
        self.s0 = SoftThreshold(theta=0.2)

        self.v1 = nn.Conv2d(out_feat, in_feat, kernel_size=3, stride=1, dilation=5, padding=5, bias=False)
        self.t1 = nn.Conv2d(in_feat, out_feat, kernel_size=3, stride=1, dilation=5, padding=5, bias=False)
        self.s1 = SoftThreshold(theta=0.2)
        
        self.v2 = nn.Conv2d(out_feat, in_feat, kernel_size=3, stride=1, dilation=5, padding=5, bias=False)
        self.t2 = nn.Conv2d(in_feat, out_feat, kernel_size=3, stride=1, dilation=5, padding=5, bias=False)
        self.s2 = SoftThreshold(theta=0.2)

    def forward(self, x):
        g = self.g(x)
        s0 = self.s0(g)

        v1 = self.v1(s0)
        t1 = self.t1(v1)
        s1 = s0 - t1 + g
        s1 = self.s1(s1)

        v2 = self.v2(s1)
        t2 = self.t2(v2)
        s2 = s1 - t2 + g
        s2 = self.s2(s2)

        return s2

class SFTLayer(nn.Module):
    def __init__(self, in_feat=32, out_feat=64):
        super(SFTLayer, self).__init__()
        self.scale_conv0 = nn.Conv2d(in_feat, in_feat, kernel_size=3, stride=1, padding=1, bias=False)
        self.scale_conv1 = nn.Conv2d(in_feat, in_feat, kernel_size=3, stride=1, padding=1, bias=False)

        self.scale_conv2 = nn.Conv2d(in_feat, in_feat, kernel_size=1, stride=1, padding=0, bias=False)
        self.scale_conv3 = nn.Conv2d(in_feat, out_feat, kernel_size=1, stride=1, padding=0, bias=False)

        self.shift_conv0 = nn.Conv2d(in_feat, in_feat, kernel_size=1, stride=1, padding=0, bias=False)
        self.shift_conv1 = nn.Conv2d(in_feat, out_feat, kernel_size=1, stride=1, padding=0, bias=False)

    def forward(self, x, side_x):
        theta = self.scale_conv1(F.leaky_relu(self.scale_conv0(side_x), 0.1, inplace=True))

        gamma = self.scale_conv3(F.leaky_relu(self.scale_conv2(theta), 0.1, inplace=True))
        beta = self.shift_conv1(F.leaky_relu(self.shift_conv0(theta), 0.1, inplace=True))
        return gamma * x + beta

class ResBlock_SFT(nn.Module):
    def __init__(self, in_feat=32, out_feat=32):
        super(ResBlock_SFT, self).__init__()
        
        self.sft = SFTLayer(in_feat=in_feat, out_feat=out_feat)
        self.block = nn.Sequential(
            nn.InstanceNorm2d(out_feat),
            nn.Conv2d(out_feat, out_feat, kernel_size=3, stride=1, padding=1, bias=False),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(out_feat, out_feat, kernel_size=3, stride=1, padding=1, bias=False)
        )
    
    def forward(self, x, side_x):
        fea = self.sft(x, side_x)
        fea = self.block(fea)
        fea = fea + x
        return fea

class ContentModule(nn.Module):
    '''a multi-scale module for content-feature extraction'''
    def __init__(self, in_feat=32, n_feat=64, epsilon=1e-4):
        super(ContentModule, self).__init__()
        assert ( n_feat % 4 == 0)
        self.feat =  n_feat // 4
        self.conv_1x1 = nn.Conv2d(in_feat, n_feat, kernel_size=1, stride=1, bias=False)
        self.conv_3x3 = nn.Conv2d(in_feat, self.feat, kernel_size=3, stride=1, padding=1, bias=False)

        self.level1_1 = nn.Conv2d(n_feat//2, n_feat//2, kernel_size=3, stride=1, padding=2, dilation=2, bias=False)
        self.level1_2 = nn.Conv2d(n_feat//2, n_feat//2, kernel_size=3, stride=1, padding=2, dilation=2, bias=False)
        
        self.level2_1 = nn.Conv2d(self.feat, self.feat, kernel_size=3, stride=1, padding=3, dilation=3, bias=False)
        self.level2_2 = nn.Conv2d(self.feat, self.feat, kernel_size=3, stride=1, padding=5, dilation=5, bias=False)
        self.level2_3 = nn.Conv2d(self.feat, self.feat, kernel_size=3, stride=1, padding=3, dilation=3, bias=False)
        self.level2_4 = nn.Conv2d(self.feat, self.feat, kernel_size=3, stride=1, padding=5, dilation=5, bias=False)
        
        self.conv_fuse = nn.Conv2d(self.feat*9, n_feat//2, kernel_size=1, stride=1, bias=False)

    def forward(self, x):
        x_ = self.conv_1x1(x)

        x1 = F.leaky_relu(self.level1_1(x_[:, :self.feat*2, :, :]), 0.1, inplace=True)
        x2 = F.leaky_relu(self.level1_2(x_[:, self.feat*2:, :, :]), 0.1, inplace=True)

        x2_1 = F.leaky_relu(self.level2_1(x1[:, :self.feat, :, :]), 0.1, inplace=True)
        x2_2 = F.leaky_relu(self.level2_2(x1[:, self.feat:, :, :]), 0.1, inplace=True)
        x2_3 = F.leaky_relu(self.level2_3(x2[:, :self.feat, :, :]), 0.1, inplace=True)
        x2_4 = F.leaky_relu(self.level2_4(x2[:, self.feat:, :, :]), 0.1, inplace=True)
 
        x = torch.cat([x1, x2, x2_1, x2_2, x2_3, x2_4, self.conv_3x3(x)], dim=1)
        x = self.conv_fuse(x)
        return x 


class UpSampling(nn.Module):
    '''feature upsampling by pixel_shuffle'''
    def __init__(self, scale=4, n_feat=32, out_feat=3):
        super(UpSampling, self).__init__()
        self.scale = scale
        
        if self.scale == 4:
            self.up_conv_1 = nn.Conv2d(n_feat, n_feat*4, kernel_size=3, stride=1, padding=1, bias=False)
            self.up_conv_2 = nn.Conv2d(n_feat, n_feat*4, kernel_size=3, stride=1, padding=1, bias=False)
            self.pixel_shuffle = nn.PixelShuffle(2)
        else: # scale 2 or 3
            c_feat = n_feat * scale * scale
            self.up_conv_1 = nn.Conv2d(n_feat, c_feat, kernel_size=3, stride=1, padding=1, bias=False)
            self.pixel_shuffle = nn.PixelShuffle(scale)

        self.conv_last = nn.Conv2d(n_feat, out_feat, kernel_size=3, stride=1, padding=1, bias=False) 
    
    def forward(self, x):
        if self.scale == 4:
            x = self.up_conv_1(x)
            x = F.relu(self.pixel_shuffle(x), inplace=True)
            x = self.up_conv_2(x)
            x = self.pixel_shuffle(x)
        else:
            x = self.up_conv_1(x)
            x = self.pixel_shuffle(x)

        return self.conv_last(x)

class SPSR(nn.Module):
    def __init__(self, args):
        super(SPSR, self).__init__()
        self.args = args
        self.scale = args.upscaling_factor
        
        self.gaussian_conv = get_gaussian_kernel() # c=3, kernel_size=3, dilation=7, sigmma=1.3
        self.sparse_prior = SparseBlock()      # c=32

        self.conv = nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1)

        self.sft_branch0 = ResBlock_SFT()
        self.sft_branch1 = ResBlock_SFT()
        self.sft_branch2 = ResBlock_SFT()
        self.sft_branch3 = ResBlock_SFT()
        self.sft_branch4 = ResBlock_SFT()
        self.sft_branch5 = ResBlock_SFT()
        self.content = ContentModule()    # c=32
        self.up_conv = UpSampling(scale=self.scale) # c=3

    def forward(self, x):
        content_x = self.gaussian_conv(x)
        detail_x = x - content_x
        detail_x = self.sparse_prior(detail_x)

        x1 = self.conv(x)

        x_ = self.sft_branch0(x1, detail_x)
        x_ = self.sft_branch1(x_, detail_x)
        x_ = self.sft_branch2(x_, detail_x)
        x_ = self.sft_branch3(x_, detail_x)
        x_ = self.sft_branch4(x_, detail_x)
        x_ = self.sft_branch5(x_, detail_x)

        x_ = self.content(x_)
        x_ = self.up_conv(x_)
        x = F.interpolate(x, scale_factor=self.scale, mode='bilinear', align_corners=False)
        return x + x_

if __name__ == "__main__":
    x = torch.ones(1, 3, 16, 16)
    print(x)
    net = SPSR()
    output = net(x)
    print("============")
    print(output)    
