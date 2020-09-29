import argparse
import random

import os
import time
import cv2
from PIL import Image

import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.utils.data
import torchvision.transforms as transforms
from torchvision.utils import make_grid
from torch.autograd import Variable
from tensorboardX import SummaryWriter

##################################################################
# from dataset import UnalignedDataset
from model_stylegan2 import StyleGAN2
##################################################################


def train(log_dir, device, gpu_ids, train_loader, num_epoch, num_epoch_resume, save_epoch_freq,
          batch_size, n_sample, lr, r1, path_regularize, path_batch_shrink, 
          g_reg_every, d_reg_every, mixing, mode_train):

    model = StyleGAN2(log_dir=log_dir, device=device, gpu_ids=gpu_ids, batch_size=batch_size, n_sample=n_sample, 
                lr=lr, r1=r1, path_regularize=path_regularize, path_batch_shrink=path_batch_shrink, 
                g_reg_every=g_reg_every, d_reg_every=d_reg_every, mixing=mixing, mode_train=mode_train)

    if num_epoch_resume != 0:
        model.log_dir = log_dir
        print('load model: epoch {}...'.format(num_epoch_resume))
        model.load('epoch_' + str(num_epoch_resume))

    writer = SummaryWriter(log_dir)

    for epoch in range(num_epoch):
        print('epoch {} started'.format(epoch + 1 + num_epoch_resume))
        t1 = time.perf_counter()

        losses = model.train(train_loader)

        t2 = time.perf_counter()
        get_processing_time = t2 - t1

        print('epoch: {}, elapsed_time: {} sec losses: {}'.format(
            epoch + 1 + num_epoch_resume, get_processing_time, losses))

        writer.add_scalar('loss_d', losses[0], epoch + 1 + num_epoch_resume)
        writer.add_scalar('loss_r1', losses[1], epoch + 1 + num_epoch_resume)
        writer.add_scalar('loss_g', losses[2], epoch + 1 + num_epoch_resume)
        writer.add_scalar('loss_path', losses[3], epoch + 1 + num_epoch_resume)
        writer.add_scalar('path_length', losses[4], epoch + 1 + num_epoch_resume)
        writer.add_scalar('mean_path_length', losses[5], epoch + 1 + num_epoch_resume)

        if (epoch + 1 + num_epoch_resume) % save_epoch_freq == 0:
            model.save('epoch_' + str(epoch + 1 + num_epoch_resume))


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='StyleGAN2 trainer')

    parser.add_argument('p', type)




    device = 'cuda:0'

    gpu_id = [0, 1, 2, 3]
    

    # random seeds
    torch.manual_seed(1234)
    np.random.seed(1234)
    random.seed(1234)

    # image
    height = 128
    width = 256

    # training details
    batch_size = 9
    lr = 0.0002  # initial learning rate for adam
    beta1 = 0.5  # momentum term of adam

    window_size = 48  # 48
    step_size = 8  # 8

    num_epoch = 100
    num_epoch_resume = 0
    save_epoch_freq = 1

    # weights of loss function
    # lambda_idt = 0.5
    # lambda_A = 10.0
    # lambda_B = 10.0
    # lambda_mask = 10.0
    lambda_idt = 5.0
    lambda_A = 10.0
    lambda_B = 10.0
    lambda_mask = 0.0

    # files, dirs
    log_dir = 'logs_3'

    # gpu
    device = torch.device("cuda:0" if torch.cuda.is_available else "cpu")
    print('device {}'.format(device))

    # dataset
    train_dataset = UnalignedDataset(batch_size, window_size, is_train=True)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    # train
    train(log_dir, device, lr, beta1, lambda_idt, lambda_A, lambda_B, lambda_mask,
          num_epoch, num_epoch_resume, save_epoch_freq)



