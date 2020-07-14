import os
import logging
import time
import utils
from collections import OrderedDict
from options import opt as args
from arch.spsr import SPSR
import skimage.color as color

import torch
import torch.nn as nn
import cv2
import numpy as np 

if __name__ == "__main__":
    print(args)
    utils.logger_info('SPSR-track', log_path='./log/SPSR_AIM2020-track.log')
    logger = logging.getLogger('SPSR-track')

    lr_path = args.data_test

    sr_path = args.result_save_dir
    if not os.path.exists(sr_path):
        os.makedirs(sr_path)

    torch.cuda.current_device()
    torch.cuda.empty_cache()
    # torch.backends.cudnn.benchmark = True

    print("===> Building model")
    net = SPSR(args)
    
    print("===> Setting GPU")
    devices_ids = list(range(args.n_gpus))
    cuda = args.gpu_mode
    if cuda and not torch.cuda.is_available():
        raise Exception("No GPU found, please run without --cuda")
    net = nn.DataParallel(net, device_ids=devices_ids).cuda()

    print("===> Loading pre-trained model")
    checkpoint = torch.load(args.resume_dir)
    if isinstance(net, torch.nn.DataParallel):
        net.module.load_state_dict(checkpoint["net"], strict=True)
    else:
        net.load_state_dict(checkpoint["net"], strict=True)

    # number of parameters
    number_parameters = sum(map(lambda x: x.numel(), net.parameters()))
    logger.info('Params number: {}'.format(number_parameters))
    
    # record PSNR, runtime
    test_results = OrderedDict()
    test_results['runtime'] = []

    logger.info(lr_path)
    logger.info(sr_path)
    idx = 0

    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)

    filelist = utils.get_list(lr_path)
    # print(len(filelist))
    # print(filelist)
    psnr_list = np.zeros(len(filelist))
    ssim_list = np.zeros(len(filelist))

    for img in filelist:
        idx += 1
        img_name, ext = os.path.splitext(os.path.basename(img))
        logger.info('{:->4d}--> {:>10s}'.format(idx, img_name+ext))
        lr = utils.imread_uint(img)
        lr = utils.uint2single(lr)
        lr = utils.single2tensor4(lr)
        lr = lr.cuda()

        with torch.no_grad():
            start.record()
            sr = net(lr)
            end.record()
            torch.cuda.synchronize()
            test_results['runtime'].append(start.elapsed_time(end))  # milliseconds

        sr_img = utils.tensor2np(sr)

        saving_path = os.path.join(sr_path, img_name+ext)
        cv2.imwrite(saving_path, sr_img[:, :, [2, 1, 0]])

    avg_runtime = sum(test_results['runtime']) / len(test_results['runtime']) / 1000.0
    logger.info('------> Average runtime of ({}) is : {:.6f} seconds'.format(lr_path, avg_runtime))
