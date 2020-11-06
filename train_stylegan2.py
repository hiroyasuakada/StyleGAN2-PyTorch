"""
usage:
    base:
        python3 -m torch.distributed.launch --nproc_per_node=N_GPU --master_port=PORT train.py --batch BATCH_SIZE LMDB_PATH
        kill $(ps aux | grep (YOUR SCRIPT NAME.py) | grep -v grep | awk '{print $2}')

    for author:
        python3 -m torch.distributed.launch --nproc_per_node=4 train_stylegan2.py lmdb_ffhq_r256_70000 --path_log logs_b4_lr0.0005
        kill $(ps aux | grep train_stylegan2.py | grep -v grep | awk '{print $2}')
"""

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
from tqdm import tqdm
import wandb

from dataset import MultiResolutionDataset, data_sampler
from model_stylegan2 import StyleGAN2

from distributed import (
    get_rank,
    synchronize,
    reduce_loss_dict,
    reduce_sum,
    get_world_size,
)


def train(log_dir, device, distributed, local_rank, train_loader, num_epoch, load_epoch, save_freq, 
          batch_size, n_sample, img_size, lr, r1, path_regularize, path_batch_shrink, 
          g_reg_every, d_reg_every, mixing):

    model = StyleGAN2(
        log_dir=log_dir, 
        device=device, 
        distributed=distributed,
        local_rank=local_rank,
        load_epoch=load_epoch,
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

    # show progress bar
    pbar1 = range(num_epoch)

    if get_rank() == 0:
        pbar1 = tqdm(
            pbar1, 
            initial=load_epoch, 
            dynamic_ncols=True, 
            smoothing=0.01, 
            unit='epoch',
            position=1
        )

    for epoch in pbar1:

        current_training_epoch = epoch + 1 + load_epoch

        running_loss_epoch = model.train(current_training_epoch, train_loader)

        if get_rank() == 0:

            # generate images during training
            with torch.no_grad():
                imgs = model.generate_imgs(
                    "epoch_" + str(current_training_epoch), 
                    truncation_target=8, 
                    truncation_rate=0.7, 
                    truncation_latent=None,
                    return_imgs=True
                )

            # save models
            if current_training_epoch % save_freq == 0:
                model.save('epoch_' + str(current_training_epoch))

            # set description for pbar1
            pbar1.set_description(
                (
                    f'd_e: {running_loss_epoch[0]:.4f}; ' 
                    f'r1_e: {running_loss_epoch[1]:.4f}; '
                    f'g_e: {running_loss_epoch[2]:.4f}; ' 
                    f'path_e: {running_loss_epoch[3]:.4f}; '
                    f'path_lengths_e: {running_loss_epoch[4]:.4f}; '
                    f'mean path_e: {running_loss_epoch[5]:.4f}; '
                )
            )

            # record logs at an epoch level
            wandb.log(
                {
                    'd_adv_loss_epoch': running_loss_epoch[0],
                    'r1_loss_epoch': running_loss_epoch[1],
                    'g_adv_loss_epoch': running_loss_epoch[2],
                    'path_loss_epoch': running_loss_epoch[3],
                    'path_lengths_epoch': running_loss_epoch[4],
                    'mean_path_length_epoch': running_loss_epoch[5],
                    'generated_images_epoch': [wandb.Image(imgs, caption='epoch: {}'.format(current_training_epoch))],
                    'epoch finished': current_training_epoch
                }
            )




if __name__ == '__main__':

    device = 'cuda'

    parser = argparse.ArgumentParser(description='StyleGAN2 trainer')

    parser.add_argument(
        '--path_dataset', type=str, default='lmdb_new', help='path to your lmdb dataset'
    )
    parser.add_argument(
        '--path_log', type=str, default='checkpoint_2', help='path to dict where log of training details will be saved'
    )
    parser.add_argument(
        '--num_epoch', type=int, default=400, help='total training epochs'
    )
    parser.add_argument(
        '--load_epoch', type=int, default=0, help='epochs to resume training '
    )
    parser.add_argument(
        '--save_freq', type=int, default=1, help='frequency of saving log'
    )
    parser.add_argument(
        '--batch_size', type=int, default=4, help='batch_size for each gpu'
    )
    parser.add_argument(
        '--lr', type=int, default=0.0005, help='learning rate'  # if 8 gpus, lr=0.002 and batch=32
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
        sampler=data_sampler(dataset, shuffle=True, distributed=args.distributed),
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
        num_epoch=args.num_epoch,
        load_epoch=args.load_epoch,
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