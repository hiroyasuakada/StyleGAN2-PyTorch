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
from tqdm import tqdm

from dataset import MultiResolutionDataset, data_sampler
from model_stylegan2_distributed import StyleGAN2

from distributed import (
    get_rank,
    synchronize,
    reduce_loss_dict,
    reduce_sum,
    get_world_size,
)


def train(log_dir, device, distributed, local_rank, train_loader, num_epoch, start_epoch, save_freq, 
          batch_size, n_sample, img_size, lr, r1, path_regularize, path_batch_shrink, 
          g_reg_every, d_reg_every, mixing):

    model = StyleGAN2(
        log_dir=log_dir, 
        device=device, 
        distributed=distributed,
        local_rank=local_rank,
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
        )

    # create a directory for logs
    if not os.path.exists(log_dir):
        os.mkdir(log_dir)

    # resume training
    if start_epoch != 0:
        model.log_dir = log_dir
        print('load model: epoch {}...'.format(start_epoch))
        model.load('epoch_' + str(start_epoch))

    # show progress bar
    pbar = range(num_epoch)
    if get_rank() == 0:
        pbar = tqdm(pbar, intial=start_epoch, dynamic_ncols=True, smoothing=0.01)

    for epoch in pbar:
        print('epoch {} started'.format(epoch + 1 + start_epoch))
        t1 = time.perf_counter()

        loss_val_mean = model.train(train_loader)

        t2 = time.perf_counter()
        get_processing_time = t2 - t1

        print('epoch: {}, elapsed_time: {:.2f}} sec losses: {}'.format(
            epoch + 1 + start_epoch, get_processing_time, losses))

        if get_rank() == 0:
            # set description for pbar
            pbar.set_description(
                (
                    f'd: {loss_val_mean[0]:.4f}; ' 
                    f'r1: {loss_val_mean[1]:.4f}; '
                    f'g: {loss_val_mean[2]:.4f}; ' 
                    f'path: {loss_val_mean[3]:.4f}; '
                    f'path_lengths: {loss_val_mean[4]:.4f}; '
                    f'mean path: {loss_val_mean[5]:.4f}; '
                )
            )

            # record logs       
            wandb.log(
                {
                    'd_adv_loss': loss_val_mean[0],
                    'r1_loss': loss_val_mean[1],
                    'g_adv_loss': loss_val_mean[2],
                    'path_loss': loss_val_mean[3],
                    'path_lengths': loss_val_mean[4],
                    'mean_path_length': loss_val_mean[5],
                }
            )

            # generate images during training
            with torch.no_grad():
                model.generate_imgs("epoch_" + str(epoch + 1 + start_epoch))

            # save models
            if (epoch + 1 + start_epoch) % save_freq == 0:
                model.save('epoch_' + str(epoch + 1 + start_epoch))


if __name__ == '__main__':

    device = 'cuda'

    parser = argparse.ArgumentParser(description='StyleGAN2 trainer')

    parser.add_argument(
        'path_dataset', type=str, help='path to your dataset, ex LMDB dataset'
    )
    parser.add_argument(
        '--path_log', type=str, default='logs', help='path to log of training details'
    )
    parser.add_argument(
        '--gpu_ids', nargs='+', type=int, default=[0, 1, 2, 3], help='GPU IDs for training'
    )
    parser.add_argument(
        '--epoch', type=int, default=100, help='total training epochs'
    )
    parser.add_argument(
        '--start_epoch', type=int, default=0, help='epochs to resume training '
    )
    parser.add_argument(
        '--save_freq', type=int, default=1, help='frequency of saving log'
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

    parser.add_argument(
        "--local_rank", type=int, default=0, help="local rank for distributed training"
    )

    args = parser.parse_args()

    # random seeds
    torch.manual_seed(1234)
    np.random.seed(1234)
    random.seed(1234)

    # configs of distributed computing
    n_gpu = int(os.environ["WORLD_SIZE"]) if "WORLD_SIZE" in os.environ else 1
    args.distributed = n_gpu > 1

    if args.distributed:
        torch.cuda.set_device(args.local_rank)
        torch.distributed.init_process_group(backend="nccl", init_method="env://")
        synchronize()

    # prepare dataset
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
        sampler=data_sampler(dataset, shuffle=True, distributed=args.distributed)
        pin_memory=True,
        drop_last=True,
    )

    if get_rank() == 0:
        wandb.init(project="stylegan2-by-PyTroch")

    # train
    train(
        device=device, 
        distributed=args.distributed,
        local_rank=args.local_rank, 
        log_dir=args.path_log, 
        train_loader=train_loader, 
        num_epoch=args.epoch,
        start_epoch=args.start_epoch,
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
        )

"""
Tips for faster training:
    https://qiita.com/sugulu_Ogawa_ISID/items/62f5f7adee083d96a587

"""