import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import math

class ChaLoss(nn.Module):
    """L1 loss with Charbonnier penalty function"""
    def __init__(self, eps=1e-3):
        super(ChaLoss, self).__init__()
        self.eps = eps
    
    def forward(self, x, y):
        diff = x - y
        return torch.sum(torch.sqrt(diff * diff + self.eps * self.eps))

class TVLoss(nn.Module):
    """Total Variation loss"""
    def __init__(self, weight=1):
        super(TVLoss, self).__init__()
        self.weight = weight

    def forward(self, x):
        # input: tensor, [n, c, h, w]
        batch_size = x.size()[0]
        h_x, w_x = x.size()[2:]
        count_h = self._tensor_size(x[:,:,1:,:])
        count_w = self._tensor_size(x[:,:,:,1:])
        h_tv = torch.pow((x[:,:,1:,:]-x[:,:,:h_x-1,:]),2).sum()
        w_tv = torch.pow((x[:,:,:,1:]-x[:,:,:,:w_x-1]),2).sum()
        loss = self.weight * 2 * (h_tv/count_h + w_tv/count_w) / batch_size
        return loss
    def _tensor_size(self, t):
        return t.size()[1] * t.size()[2] * t.size()[3]

# MS-SSIM
def gaussian_kernel(window_size, sigma):
    kernel = torch.Tensor([math.exp(-(x - window_size//2)**2 / float(2*sigma**2)) for x in range(window_size)])
    return kernel / kernel.sum()

def create_window(window_size, sigma, channel):
    _1D_window = gaussian_kernel(window_size, sigma).unsqueeze(1)
    _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
    window = Variable(_2D_window.expand(channel, 1, window_size, window_size).contiguous())
    return window

class MS_SSIM(nn.Module):
    def __init__(self, size_average=True, rgb_range=255):
        super(MS_SSIM, self).__init__()
        self.size_average = size_average
        self.channel = 3
        self.rgb_range = rgb_range
    
    def _ssim(self, img1, img2, size_average = True):

        _, c, w, h = img1.size()
        window_size = min(w, h, 11)
        sigma = 1.5 * window_size / 11
        window = create_window(window_size, sigma, self.channel).cuda()
        mu1 = F.conv2d(img1, window, padding = window_size//2, groups = self.channel)
        mu2 = F.conv2d(img2, window, padding = window_size//2, groups = self.channel)

        mu1_sq = mu1.pow(2)
        mu2_sq = mu2.pow(2)
        mu1_mu2 = mu1*mu2

        sigma1_sq = F.conv2d(img1*img1, window, padding = window_size//2, groups = self.channel) - mu1_sq
        sigma2_sq = F.conv2d(img2*img2, window, padding = window_size//2, groups = self.channel) - mu2_sq
        sigma12 = F.conv2d(img1*img2, window, padding = window_size//2, groups = self.channel) - mu1_mu2

        C1 = (0.01*self.rgb_range)**2
        C2 = (0.03*self.rgb_range)**2
        V1 = 2.0 * sigma12 + C2
        V2 = sigma1_sq + sigma2_sq + C2
        ssim_map = ((2*mu1_mu2 + C1)*V1)/((mu1_sq + mu2_sq + C1)*V2)
        mcs_map = V1 / V2
        if size_average:
            return ssim_map.mean(), mcs_map.mean()

    def ms_ssim(self, img1, img2, levels=5):

        weight = Variable(torch.Tensor([0.0448, 0.2856, 0.3001, 0.2363, 0.1333]).cuda())

        msssim = Variable(torch.Tensor(levels,).cuda())
        mcs = Variable(torch.Tensor(levels,).cuda())
        for i in range(levels):
            ssim_map, mcs_map = self._ssim(img1, img2)
            msssim[i] = ssim_map
            mcs[i] = mcs_map
            filtered_im1 = F.avg_pool2d(img1, kernel_size=2, stride=2)
            filtered_im2 = F.avg_pool2d(img2, kernel_size=2, stride=2)
            img1 = filtered_im1
            img2 = filtered_im2

        value = (torch.prod(mcs[0:levels-1]**weight[0:levels-1])*
                                    (msssim[levels-1]**weight[levels-1]))
        return value


    def forward(self, img1, img2):

        return self.ms_ssim(img1, img2)

class LTLoss(nn.Module):
    '''L1 + TV'''
    def __init__(self):
        super(LTLoss, self).__init__()
        self.l1 = nn.L1Loss()
        self.tv = TVLoss(weight=1e-4)
    
    def forward(self, output, target):
        return self.l1(output, target) + self.tv(output)

class CTLoss(nn.Module):
    '''L1 with Charbonnier penalty + TV'''
    def __init__(self):
        super(CTLoss, self).__init__()
        self.cl = ChaLoss()
        self.tv = TVLoss(weight=1e-3)
    
    def forward(self, output, target):
        return self.cl(output, target) + self.tv(output)

class LMLoss(nn.Module):
    '''L1 + MS-SSIM'''
    def __init__(self):
        super(LMLoss, self).__init__()
        self.l1 = nn.L1Loss()
        self.ms_ssim = MS_SSIM(rgb_range=1)
    
    def forward(self, output, target):
        return 0.99 * self.l1(output, target) + (1-0.99) * self.ms_ssim(output, target)


