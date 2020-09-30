import argparse
import random

import os
import time
import cv2
from PIL import Image

import numpy as np
import torch
import torch.nn as nn
from torch.utils import data
from torchvision import transforms
from torchvision.utils import make_grid
from tensorboardX import SummaryWriter

from dataset import MultiResolutionDataset
from model_stylegan2 import StyleGAN2


def train(log_dir, device, gpu_ids, train_loader, num_epoch, num_epoch_resume, save_freq, 
          batch_size, n_sample, img_size, lr, r1, path_regularize, path_batch_shrink, 
          g_reg_every, d_reg_every, mixing, mode_train):

    model = StyleGAN2(
        log_dir=log_dir, 
        device=device, 
        gpu_ids=gpu_ids, 
        batch_size=batch_size, 
        n_sample=n_sample, 
        img_size=img_size,
        lr=lr, 
        r1=r1, 
        path_regularize=path_regularize, 
        path_batch_shrink=path_batch_shrink, 
        g_reg_every=g_reg_every, 
        d_reg_every=d_reg_every, 
        mixing=mixing, 
        mode_train=mode_train)

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

        if (epoch + 1 + num_epoch_resume) % save_freq == 0:
            model.save('epoch_' + str(epoch + 1 + num_epoch_resume))

    writer.export_scalars_to_json("./all_scalars.json")
    writer.close()
    ### tensorboard --logdir runs


if __name__ == '__main__':

    device = 'cuda:0'

    parser = argparse.ArgumentParser(description='StyleGAN2 trainer')

    parser.add_argument(
        'path_log', type=str, help='path to log of training details'
    )
    parser.add_argument(
        'path_dataset', type=str, help='path to your dataset, ex LMDB dataset'
    )
    parser.add_argument(
        '--gpu_ids', nargs='+', type=int, default=[0, 1, 2, 3], help='GPU IDs for training'
    )
    parser.add_argument(
        '--epoch', type=int, default=100, help='total training epochs'
    )
    parser.add_argument(
        '--epoch_resume', type=int, default=0, help='epochs to resume training '
    )
    parser.add_argument(
        'save_freq', type=int, default=1, help='frequency of saving log'
    )
    parser.add_argument(
        '--batch_size', type=int, default=16, help='batch_size for training'
    )
    parser.add_argument(
        '--lr', type=int, default=0.001, help='learning rate'  # if 8 gpus, lr=0.002 and batch=32
    )
    parser.add_argument(
        '--n_sample', type=int, default=16, help ='the number of imgs generated during training'
    )
    parser.add_argument(
        '--img_size', type=int, default=256, help='image sizes for the model'
    )

    parser.add_argument(
        '--r1', type=float, default=10, help='weight of the r1 regularization'
    )
    parser.add_argument(
        '--path_regularize', type=float, default=2, help='weight of the path length regularization'
    )
    parser.add_argument(
        '--path_batch_shrink', 
        type=int, 
        default=2, 
        help='batch size reducing factor for the path length regularization'
    )
    parser.add_argument(
        "--d_reg_every",
        type=int,
        default=16,
        help="interval of the applying r1 regularization",
    )
    parser.add_argument(
        "--g_reg_every",
        type=int,
        default=4,
        help="interval of the applying path length regularization",
    )
    parser.add_argument(
        "--mixing", type=float, default=0.9, help="probability of latent code mixing"
    )

    args = parser.parse_args()

    # random seeds
    torch.manual_seed(1234)
    np.random.seed(1234)
    random.seed(1234)

    # dataset
    transform = transforms.Compose(
        [
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5), inplace=True),
        ]
    )
    dataset = MultiResolutionDataset(
        path=args.path_dataset, 
        transform=transform, 
        resolution=args.img_size, 
    )
    train_loader = data.DataLoader(
        dataset, 
        batch_size=args.batch_size, 
        num_workers=os.cpu_count(),
        pin_memory=True,
        shuffle=True,
        drop_last=True,
    )

    # train
    train(device=device, 
          log_dir=args.path_log, 
          gpu_ids=args.gpu_ids, 
          train_loader=train_loader, 
          num_epoch=args.num_epoch,
          num_epoch_resume=args.num_epoch_resume,
          save_freq=args.save_freq, 
          batch_size=args.batch_size, 
          n_sample=args.n_sample, 
          img_size=args.img_size,
          lr=args.lr, 
          r1=args.r1, 
          path_regularize=args.path_regularize, 
          path_batch_shrink=args.path_batch_shrink, 
          g_reg_every=args.g_reg_every, 
          d_reg_every=args.d_reg_every, 
          mixing=args.mixing, 
          mode_train=True,
          )

"""
Tips for faster training:
    https://qiita.com/sugulu_Ogawa_ISID/items/62f5f7adee083d96a587

"""