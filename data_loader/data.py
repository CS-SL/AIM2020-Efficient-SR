import torch

import os
from os import path
from os import listdir
import glob
import numpy as np
from PIL import Image 

from torch.utils.data import Dataset
from torchvision.transforms import ToTensor

class DataGenerator(Dataset):
    def __init__(self, opt, data_dir,
                data_name = 'DIV2K', 
                train = True
                ):
        super(DataGenerator, self).__init__()

        assert path.isdir(data_dir)
        
        self.opt = opt
        self.data_dir = data_dir
        self.name = data_name
        self.train = train
        self.upscaling_factor = opt.upscaling_factor
        self.patch_size = opt.patch_size
        self.aug_flag = opt.agu_flag
        self.crop_flag = opt.crop_flag
        
        self._set_filesystem(self.data_dir)
        self.hr_list, self.lr_list = self._scan()

    def _getitem_test(self, index):
        lr_data, hr_data, filename = self._load_file(index)
        
        # to tensor
        # Convert a numpy.ndarray (H x W x C) [0, 255] to a torch.FloatTensor of shape (C x H x W) [0.0, 1.0]
        lr_data = ToTensor()(lr_data)
        hr_data = ToTensor()(hr_data)

        return lr_data, hr_data, filename

    def _getitem_train(self, index):
        index = index % len(self.lr_list)
        lr_data, hr_data, filename = self._load_file(index)

        # get image patch
        if self.crop_flag:
            ih, iw, _ = lr_data.shape

            crop_h = np.random.randint(0, ih - self.patch_size + 1)
            crop_w = np.random.randint(0, iw - self.patch_size + 1)
            
            crop_hr_h = self.upscaling_factor * crop_h
            crop_hr_w = self.upscaling_factor * crop_w
            
            patch_size_hr = self.patch_size * self.upscaling_factor

            lr_data = lr_data[crop_h:crop_h+self.patch_size, crop_w :crop_w +self.patch_size, :]
            hr_data = hr_data[crop_hr_h:crop_hr_h+patch_size_hr, crop_hr_w:crop_hr_w+patch_size_hr, :]
        
        # data agumentation
        if self.aug_flag:
            flip_rot = np.random.randint(0, 8)
            lr_data = self._data_agument(lr_data, flip_rot).copy()
            hr_data = self._data_agument(hr_data, flip_rot).copy()
        
        # to tensor
        # Convert a numpy.ndarray (H x W x C) [0, 255] to a torch.FloatTensor of shape (C x H x W) [0.0, 1.0]
        lr_data = ToTensor()(lr_data)
        hr_data = ToTensor()(hr_data)

        return lr_data, hr_data, filename

    def __getitem__(self, index):
        return self._getitem(index)
    
    def __len__(self):
        return self.len_data
    
    def _set_filesystem(self, data_dir):
        self.apath = path.join(data_dir, self.name)
        self.hr_dir = path.join(self.apath, 'DIV2K_train_HR')
        self.lr_dir = path.join(self.apath, 'DIV2K_train_LR_bicubic/X{}'.format(self.upscaling_factor))

    def _scan(self):
        names_hr = sorted(glob.glob(path.join(self.hr_dir, "*.png")))
        names_lr = sorted(glob.glob(path.join(self.lr_dir, "*.png")))
        assert len(names_hr) == len(names_lr)
        return names_hr, names_lr
    
    def _read_img(self, path):
        '''Load an image by PIL.Image.open'''
        img = Image.open(path).convert('RGB')
        img = np.array(img, dtype=np.uint8)
        return img
    
    def _load_file(self, index):
        lr_data = self._read_img(self.lr_list[index])
        hr_data = self._read_img(self.hr_list[index])
        filename = path.basename(path.splitext(self.lr_list[index])[0])
        return lr_data, hr_data, filename

    def _data_agument(self, img, flip_rot):
        # img: numpy, HWC
        assert 0 <= flip_rot <= 7
        flip_flag = flip_rot % 2
        rot_flag = flip_rot // 2

        if rot_flag > 0:
            img = np.rot90(img, k=rot_flag, axes=(0, 1))
        if flip_flag > 0:
            img = np.flip(img, axis=0)
        return img


class DIV2K_train(DataGenerator):
    def __init__(self, opt, data_dir, data_name='DIV2K', train=True):
        super(DIV2K_train, self).__init__(opt, data_dir=data_dir, data_name=data_name, train=train)
        
        if self.train:
            self._getitem = self._getitem_train
            self.len_data = len(self.lr_list) * (opt.test_every // (len(self.lr_list) // opt.batch_size))
       
    
   
class DIV2K_val(DataGenerator):
    def __init__(self, opt, data_dir, data_name='DIV2K', train=False):
        super(DIV2K_val, self).__init__(opt, data_dir=data_dir, data_name=data_name, train=train)

        self.aug_flag = False
        self.crop_flag = False
        
        if self.train == False:
            self._getitem = self._getitem_test
            self.len_data = len(self.lr_list)

    def _set_filesystem(self, data_dir):
        self.apath = path.join(data_dir, self.name, 'Val')
        self.hr_dir = path.join(self.apath, 'HR')
        self.lr_dir = path.join(self.apath, 'LR_bicubic/X{}'.format(self.upscaling_factor))

    def _scan(self):
        names_hr, names_lr = super(DIV2K_val, self)._scan()
        names_hr = names_hr[0:10]
        names_lr = names_lr[0:10]
        return names_hr, names_lr


if __name__ == "__main__":
    data_dir = '/root/proj/AIM2020/dataset/'
    dataset = 'DIV2K'
    # ds = DataGenerator(data_dir=data_dir, data_name=dataset)
    # print(len(ds))
    train_data = DIV2K_train(data_dir=data_dir, data_name=dataset)
    print(len(train_data))
    val_data = DIV2K_val(data_dir=data_dir, data_name=dataset)
    print(len(val_data))

    # for i in range(len(ds)):
    #     print('filename:{} || LR_Shape:{} || HR:{}'.format(ds[i][2], ds[i][0].shape, ds[i][1].shape))
    for i in range(len(val_data)):
        print('filename:{} || LR_Shape:{} || HR:{}'.format(val_data[i][2], val_data[i][0].shape, val_data[i][1].shape))
    
