import os
import time
import datetime
import random
import numpy as np
from tqdm import tqdm
import torch
from torch import nn
from torch.autograd import Variable
from torch import autograd
from torch.utils.data import DataLoader
from torch.backends import cudnn
from torchvision import transforms

import ssim
import dataset
import utils



def pretrain(configs):
    cudnn.benchmark = configs.cudnn_benchmark

    # configurations
    save_folder = configs.save_path
    sample_folder = configs.sample_path
    utils.check_path(save_folder)
    utils.check_path(sample_folder)

    # Loss functions
    if configs.no_gpu == False:
        criterion_L1 = torch.nn.L1Loss().cuda()
        criterion_L2 = torch.nn.MSELoss().cuda()
        criterion_ssim = ssim.SSIM().cuda()
    else: 
        criterion_L1 = torch.nn.L1Loss()
        criterion_L2 = torch.nn.MSELoss()
        criterion_ssim = ssim.SSIM()

    # Initialize Generator
    generator = utils.create_generator(configs)

    # To device
    if configs.no_gpu == False:
        if configs.multi_gpu:
            generator = nn.DataParallel(generator)
            generator = generator.cuda()
        else:
            generator = generator.cuda()

    # Optimizers
    optimizer_G = torch.optim.Adam(filter(lambda p: p.requires_grad, generator.parameters()), lr = configs.lr_g, betas = (configs.b1, configs.b2), weight_decay = configs.weight_decay)

    print("Pretrained models loaded")

    # Learning rate decrease
    def adjust_learning_rate(configs, epoch, optimizer):
        target_epoch = configs.epochs - configs.lr_decrease_epoch
        remain_epoch = configs.epochs - epoch
        if epoch >= configs.lr_decrease_epoch:
            lr = configs.lr_g * remain_epoch / target_epoch
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr
    
    # Save the model if pre_train == True
    def save_model(configs, epoch, iteration, len_dataset, generator):
        """
        Save the model at "checkpoint_interval" and its multiple
        """
        # Define the name of trained model
        model_name = ''
        if configs.save_mode == 'epoch':
            model_name = 'KPN_rainy_image_epoch%d_bs%d.pth' % (epoch, configs.train_batch_size)
        if configs.save_mode == 'iter':
            model_name = 'KPN_rainy_image_iter%d_bs%d.pth' % (iteration, configs.train_batch_size)
        save_model_path = os.path.join(configs.save_path, model_name)
        if configs.multi_gpu == True:
            if configs.save_mode == 'epoch':
                if (epoch % configs.save_by_epoch == 0) and (iteration % len_dataset == 0):
                    torch.save(generator.module.state_dict(), save_model_path)
                    print('The trained model is successfully saved at epoch %d' % (epoch))
            if configs.save_mode == 'iter':
                if iteration % configs.save_by_iter == 0:
                    torch.save(generator.module.state_dict(), save_model_path)
                    print('The trained model is successfully saved at iteration %d' % (iteration))
        else:
            if configs.save_mode == 'epoch':
                if (epoch % configs.save_by_epoch == 0) and (iteration % len_dataset == 0):
                    torch.save(generator.state_dict(), save_model_path)
                    print('The trained model is successfully saved at epoch %d' % (epoch))
            if configs.save_mode == 'iter':
                if iteration % configs.save_by_iter == 0:
                    torch.save(generator.state_dict(), save_model_path)
                    print('The trained model is successfully saved at iteration %d' % (iteration))

    gpu_num = torch.cuda.device_count()
    print("Using %d GPU(s)" % gpu_num)
    
    # Define the dataset
    trainset = dataset.DenoisingDataset(configs)
    print('The overall number of training images:', len(trainset))

    # Define the dataloader
    train_loader = DataLoader(trainset, batch_size=configs.train_batch_size, shuffle=True, num_workers=configs.num_workers, pin_memory=True)
    
    # ----------------------------------------
    #                 Training
    # ----------------------------------------

    # Count start time
    prev_time = time.time()
    # For loop training
    for epoch in range(configs.epochs):
        for i, (true_input, true_target, _, _) in enumerate(train_loader):
            if configs.no_gpu == False:
                # To device
                true_input = true_input.cuda()
                true_target = true_target.cuda()

            # Train Generator
            optimizer_G.zero_grad()
            fake_target = generator(true_input, true_input)            
            ssim_loss = -criterion_ssim(true_target, fake_target)

            Pixellevel_L1_Loss = criterion_L1(fake_target, true_target)

            # Overall Loss and optimize
            loss = Pixellevel_L1_Loss + 0.2 * ssim_loss

            loss.backward()
            optimizer_G.step()

            # Determine approximate time left
            iters_done = epoch * len(train_loader) + i
            iters_left = configs.epochs * len(train_loader) - iters_done
            time_left = datetime.timedelta(seconds = iters_left * (time.time() - prev_time))
            prev_time = time.time()

            # Print log
            print("\r[Epoch %d/%d] [Batch %d/%d] [Loss: %.4f %.4f] Time_left: %s" %
                ((epoch + 1), configs.epochs, i, len(train_loader), Pixellevel_L1_Loss.item(), ssim_loss.item(), time_left))

            # Save model at certain epochs or iterations
            save_model(configs, (epoch + 1), (iters_done + 1), len(train_loader), generator)

            # Learning rate decrease at certain epochs
            adjust_learning_rate(configs, (epoch + 1), optimizer_G)

        ### Sample data every epoch
        if (epoch + 1) % 1 == 0:
            img_list = [true_input, fake_target, true_target]
            name_list = ['in', 'pred', 'gt']
            utils.save_sample_png(sample_folder = sample_folder, sample_name = 'train_epoch%d' % (epoch + 1), img_list = img_list, name_list = name_list, pixel_max_cnt = 255)