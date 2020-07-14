import os
import time
import math
import numpy as np
import skimage.color as color
import cv2
import logging

import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.utils.data import DataLoader

from data_loader.data_npy import DIV2K_train, DIV2K_val, DF2K_train, DF2K_val
# from arch.model_gaussian import SPSR
from arch.model_gaussian_w_eca import SPSR
from options import opt as args
from loss import CTLoss
from utils import save_model, print_network, calc_psnr, calc_ssim, tensor2np, shave, quantize, save_img, tensor2uint, logger_info
from torch.utils.tensorboard import SummaryWriter

# set flags / seeds
torch.backends.cudnn.benchmark = False
np.random.seed(1)
torch.manual_seed(1)
torch.cuda.manual_seed(1)

# Start with main code
if __name__ == '__main__':
    
    print(args)
    
    tblog_dir = os.path.join(args.log_dir, 'X{}'.format(args.upscaling_factor))
    if not os.path.exists(tblog_dir):
        os.makedirs(tblog_dir)
    writer = SummaryWriter(log_dir=tblog_dir, comment='sparse')

    log_path = os.path.join(args.log_dir, 'train', 'X{}'.format(args.upscaling_factor))
    if not os.path.exists(log_path):
        os.makedirs(log_path)
    logger_info('SPSR-train-track', log_path=log_path+'/track.log')
    logger = logging.getLogger('SPSR-train-track')

    # add code for datasets
    print("===> Loading datasets")
    train_data_set = DIV2K_train(opt=args, data_dir=args.data_dir, data_name=args.data_train)
    val_data_set = DIV2K_val(opt=args, data_dir=args.data_dir, data_name=args.data_train)
    # train_data_set = DF2K_train(opt=args, data_dir=args.data_dir, data_name=args.data_train)
    # val_data_set = DF2K_val(opt=args, data_dir=args.data_dir, data_name=args.data_train)
   
    training_data = DataLoader(dataset=train_data_set, batch_size=args.batch_size, num_workers=args.n_threads, shuffle=True, pin_memory=True)
    val_data = DataLoader(dataset=val_data_set, batch_size=1, num_workers=args.n_threads, shuffle=False)
    
    # instantiate network
    print("===> Building model")
    devices_ids = list(range(args.n_gpus))
    net = SPSR(args)

    # if running on GPU and we want to use cuda move model there
    print("===> Setting GPU")
    cuda = args.gpu_mode
    if cuda and not torch.cuda.is_available():
        raise Exception("No GPU found, please run without --cuda")
    net = nn.DataParallel(net, device_ids=devices_ids)
    net = net.cuda()

    # create loss
    # cha_loss = ChaLoss()
    # l1 = nn.L1Loss()
    ctloss = CTLoss()
    
    print('---------- Networks architecture -------------')
    print_network(net)
    print('----------------------------------------------')

    # create optimizer
    print("===> Setting Optimizer")
    optimizer = torch.optim.Adam(net.parameters(), lr=args.lr, betas=(0.9, 0.999), eps=1e-8, weight_decay=args.weight_decay)

    # optionally ckp from a checkpoint
    if args.resume:
        if os.path.isfile(args.resume_dir):
            print("======> loading checkpoint at '{}'".format(args.resume))
            checkpoint = torch.load(args.resume_dir)

            if isinstance(net, torch.nn.DataParallel):
                net.module.load_state_dict(checkpoint["net"], strict=True)
            else:
                net.load_state_dict(checkpoint["net"], strict=True)

            optimizer.load_state_dict(checkpoint['optimizer'])
            args.start_epoch = checkpoint['epoch']

        else:
            print("======> founding no checkpoint at '{}'".format(args.resume_dir))
           
    print("===> Training")
    for epoch in range(args.start_epoch, args.total_epochs + 1):
        # learning rate is decayed by 2 every 200 epochs
        lr_ = args.lr * (0.5 ** (epoch // args.decay_step))
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr_
        
        print("epoch = {}".format(epoch), "lr = {}".format(optimizer.param_groups[0]["lr"]))
        
        epoch_start_time = time.time()
        epoch_loss = 0
        
        net.train()
        for iteration, batch in enumerate(training_data, 1):
            lr, hr, _ = batch[0], batch[1], batch[2]
            lr = lr.cuda()
            hr = hr.cuda()

            optimizer.zero_grad()

            t0 = time.time()
            sr = net(lr)
            loss = ctloss(sr, hr) / args.batch_size
            epoch_loss += loss.item()
            t1 = time.time()
            
            loss.backward()
            optimizer.step()

            if iteration % 100 == 0:
                logger.info("===> Epoch[{}/{}]({}/{}): lr:{} || Loss: {:.4f} || Timer: {:.4f} sec.".format(epoch, args.total_epochs, iteration, len(training_data), 
                                                                                                    optimizer.param_groups[0]['lr'], loss.item(), (t1 - t0)))
     
        # Epoch loss
        epoch_time = time.time() - epoch_start_time
        logger.info("===> Epoch{}: AVG Loss:{:.4f} || Timers:{:.3f} sec".format(epoch, epoch_loss/len(training_data), epoch_time))
        writer.add_scalar("Epoch Loss", epoch_loss / len(training_data), epoch)
        
        # checkpoint
        save_model(net, epoch, optimizer, args.snapshot_dir, args.upscaling_factor)

        # Evaluation
        if epoch % 1 == 0:
            logger.info('Evaluation....')
            net.eval()
            avg_psnr = 0
            avg_ssim = 0

            n_count, n_total = 1, len(val_data)
            for batch in val_data:
                lr, hr, filename = batch[0], batch[1], batch[2]
                lr = lr.cuda()
                hr = hr.cuda()
            
                with torch.no_grad():
                    sr = net(lr)

                sr_img = tensor2np(sr)
                gt_img = tensor2np(hr)
                
                if args.calc_y:
                    sr_img_y = quantize(color.rgb2ycbcr(sr_img)[:, :, 0])
                    gt_img_y = quantize(color.rgb2ycbcr(gt_img)[:, :, 0])
                    avg_psnr += calc_psnr(sr_img_y, gt_img_y)
                    avg_ssim += calc_ssim(sr_img_y, gt_img_y)
                else:
                    avg_psnr += calc_psnr(sr_img, gt_img)
                    avg_ssim += calc_ssim(sr_img, gt_img)

                saving_path = os.path.join(args.result_save_dir, 'X{}'.format(args.upscaling_factor))
                if not os.path.exists(saving_path):
                    os.makedirs(saving_path)
                filename = str(filename[0]) + '_SR' + '.png'
                saving_path = os.path.join(saving_path, filename)
                print("===> Processing: {}/{}".format(n_count, n_total))
                cv2.imwrite(saving_path, sr_img[:, :, [2, 1, 0]])
                n_count += 1

            avg_psnr /= len(val_data)
            avg_ssim /= len(val_data)
            logger.info("Valid_epoch [{}] || psnr: {:.4f} || ssim: {:.4f}".format(epoch, avg_psnr, avg_ssim))
            
            writer.add_scalar('avg_psnr', avg_psnr, epoch)
            writer.add_scalar('avg_ssim', avg_ssim, epoch)
            
    writer.close()   