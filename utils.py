import os
import time
import torch
import torch.nn as nn 
import cv2
import numpy as np
import math
from skimage.metrics import peak_signal_noise_ratio as psnr 
from skimage.metrics import structural_similarity as ssim

IMG_EXTENSIONS = ['.jpg', '.JPG', '.jpeg', '.JPEG', '.png', '.PNG', '.ppm', '.PPM', '.bmp', '.BMP']


def is_image_file(filename):
    return any(filename.endswith(extension) for extension in IMG_EXTENSIONS)

def print_network(net):
    num_params = 0
    for param in net.parameters():
        num_params += param.numel()
    print(net)
    print('Total number of parameters: {}'.format(num_params))


def save_model(model, epoch, optimizer, snapshot_dir, upscaling_factor):
    save_dir = os.path.join(snapshot_dir, 'X{}'.format(upscaling_factor))
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    if isinstance(model, torch.nn.DataParallel):
        model_state_dict = model.module.state_dict()
    else:
        model_state_dict = model.state_dict()

    checkpoint = {
        "net": model_state_dict,
        'optimizer': optimizer.state_dict(),
        "epoch": epoch
    }

    torch.save(checkpoint, save_dir +'/'+ 'model_{:03d}_epoch.pth'.format(epoch))
    print('The SR model is saved.')

####################
## Blur Operation ##
####################

class MeanConv(nn.Module):
    """smoothing an image via Conv-Mean Filter"""
    def __init__(self):
        super(MeanConv, self).__init__()
        self.box_filter = nn.Conv2d(in_channels=3, out_channels=3, kernel_size=3, stride=1, padding=1, bias=False, groups=3)
    def forward(self, x):
        _, _, h, w = x.size()
        N = self.box_filter(x.data.new().resize_((1, 3, h, w)).fill_(1.0))
        smooth_x = self.box_filter(x) / N
        return smooth_x

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

    gaussian_filter.weight.data = gaussian_kernel
    gaussian_filter.weight.requires_grad = False

    return gaussian_filter

#################### 
# Image Processing #
####################
def get_list(path):
    return sorted([os.path.join(path, f) for f in os.listdir(path) if f.endswith('.png')])

def imread_uint(path, n_channels=3):
    # input: path
    # output: HxWx3(RGB or GGG), or HxWx1 (G)
    if n_channels == 1:
        img = cv2.imread(path, 0)  # cv2.IMREAD_GRAYSCALE
        img = np.expand_dims(img, axis=2)  # HxWx1
    elif n_channels == 3:
        img = cv2.imread(path, cv2.IMREAD_UNCHANGED)  # BGR or G
        if img.ndim == 2:
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)  # GGG
        else:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # RGB
    return img

def uint2single(img):
    return np.float32(img/255.)

# convert uint (HxWxn_channels) to 4-dimensional torch tensor
def single2tensor4(img):
    return torch.from_numpy(np.ascontiguousarray(img)).permute(2, 0, 1).float().unsqueeze(0)

def save_img(img, img_name, result_save_dir, upscaling_factor):
    save_img = img.squeeze().float().clamp_(0, 1).cpu()
    save_img = (save_img * 255).numpy()
    # print(save_img.shape)
    # save img
    save_dir=os.path.join(result_save_dir, 'X{}'.format(upscaling_factor))
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
        
    img_name = img_name + '.png'
    save_fn = os.path.join(save_dir, img_name)
    cv2.imwrite(save_fn, save_img)
    print('Saving!')

def tensor2np(tensor, out_type=np.uint8, min_max=(0, 1)):
    tensor = tensor.data.squeeze().float().cpu().clamp_(*min_max)
    tensor = (tensor - min_max[0]) / (min_max[1] - min_max[0])  # to range [0, 1]
    img_np = tensor.numpy()
    img_np = np.transpose(img_np, (1, 2, 0))
    if out_type == np.uint8:
        img_np = (img_np * 255.0).round()

    return img_np.astype(out_type)
    
def tensor2uint(img):
    img = img.data.squeeze().float().clamp_(0, 255).cpu().numpy()
    if img.ndim == 3:
        img = np.transpose(img, (1, 2, 0))
    return np.uint8((img).round())

def modcrop(img, scale):
    # img_in: Numpy, HWC or HW
    img = np.copy(img)
    if img.ndim == 2:
        H, W = img.shape
        H_r, W_r = H % scale, W % scale
        img = img[:H - H_r, :W - W_r]
    elif img.ndim == 3:
        H, W, C = img.shape
        H_r, W_r = H % scale, W % scale
        img = img[:H - H_r, :W - W_r, :]
    else:
        raise ValueError('Wrong img ndim: [{:d}].'.format(img.ndim))
    return img

def shave(img, border=0):
    # img: numpy, HWC or HW
    img = np.copy(img)
    h, w = img.shape[:2]
    img = img[border:h-border, border:w-border]
    return img

def quantize(img):
    return img.clip(0, 255).round().astype(np.uint8)


###############################
#### metrics: PSNR, SSIM ######
###############################
def calc_psnr(im1, im2):
    psnr_val = psnr(im1, im2)
    return psnr_val

def calc_ssim(im1, im2):
    isRGB = len(im1.shape) == 3 and im1.shape[-1] == 3
    ssim_val = ssim(im1, im2, K1=0.01, K2=0.03, gaussian_weights=True, sigma=1.5, use_sample_covariance=False,
             multichannel=isRGB)
    return ssim_val


########## 
# logger #
##########

import os
import sys
import datetime
import logging

def log(*args, **kwargs):
    print(datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S:"), *args, **kwargs)

def logger_info(logger_name, log_path='default_logger.log'):
    log = logging.getLogger(logger_name)
    if log.hasHandlers():
        print('LogHandlers exist!')
    else:
        print('LogHandlers setup!')
        level = logging.INFO
        formatter = logging.Formatter('%(asctime)s.%(msecs)03d : %(message)s', datefmt='%y-%m-%d %H:%M:%S')
        fh = logging.FileHandler(log_path, mode='a')
        fh.setFormatter(formatter)
        log.setLevel(level)
        log.addHandler(fh)
        # print(len(log.handlers))

        sh = logging.StreamHandler()
        sh.setFormatter(formatter)
        log.addHandler(sh)

# print to file and std_out simultaneously
class logger_print(object):
    def __init__(self, log_path="default.log"):
        self.terminal = sys.stdout
        self.log = open(log_path, 'a')

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)  # write the message

    def flush(self):
        pass
