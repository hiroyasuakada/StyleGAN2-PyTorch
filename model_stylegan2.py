import time
import random
import math
import os
import wandb
from PIL import Image

import numpy as np
import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import autograd
from torch.utils import data
import torch.distributed as dist
from torchvision import utils 
from tqdm import tqdm
import wandb

from model_networks import Generator, Discriminator
from dataset import Dataset

from distributed import (
    get_rank,
    synchronize,
    reduce_loss_dict,
    reduce_sum,
    get_world_size,
)


###################################################################################################
###                                         StyleGAN2                                           ###
###################################################################################################


class StyleGAN2(object):
    def __init__(self, log_dir='logs', device='cuda', distributed=True, local_rank=0, load_epoch=0, 
                 batch_size=16, n_sample=16, img_size=1024, lr=0.001, r1=10, 
                 path_regularize=2, path_batch_shrink=2, 
                 g_reg_every=4, d_reg_every=16, mixing=0.9):
        
        # faster training
        torch.backends.cudnn.benchmark = True
        
        self.batch_size = batch_size
        self.n_sample = n_sample
        self.img_size = img_size
        self.log_dir = log_dir
        self.device = device
        # print(torch.cuda.is_available())
        self.distributed = distributed
        self.local_rank = local_rank
        self.load_epoch = load_epoch
        
        self.r1 = r1
        self.lr = lr
        self.g_reg_every = g_reg_every
        self.d_reg_every = d_reg_every
        self.latent_size = 512
        self.path_regularize = path_regularize
        self.path_batch_shrink = path_batch_shrink
        self.mixing = mixing

        # prepare samples to generate images during training
        self.sample_z = torch.randn(self.n_sample, self.latent_size, device=self.device)
    
        # initialize loss functions and so on
        self.r1_loss = torch.tensor(0.0, device=self.device)
        self.path_loss = torch.tensor(0.0, device=self.device)
        self.path_lengths = torch.tensor(0.0, device=self.device)
        self.mean_path_length = 0
        self.mean_path_length_avg = 0
        self.loss_dict = {}

        # # load epoch 219
        # self.r1_loss = torch.tensor(0.03054, device=self.device)
        # self.path_loss = torch.tensor(0.01291, device=self.device)
        # self.path_lengths = torch.tensor(0.2501, device=self.device)
        # self.mean_path_length = 0.3633
        # self.mean_path_length_avg = 0.3633

        # load networks
        self.G = Generator(resolution=self.img_size, latent_size=self.latent_size).to(self.device)
        self.D = Discriminator(resolution=self.img_size).to(self.device)

        self.G.train()
        self.D.train()

        # optimize params for G and D
        g_reg_ratio = g_reg_every / (g_reg_every + 1)
        d_reg_ratio = d_reg_every / (d_reg_every + 1)

        self.optimizer_G = torch.optim.Adam(
            self.G.parameters(), 
            lr=self.lr * g_reg_ratio, 
            betas=(0 ** g_reg_ratio, 0.99 ** g_reg_ratio)
        )
        self.optimizer_D = torch.optim.Adam(
            self.D.parameters(), 
            lr=self.lr * d_reg_ratio, 
            betas=(0 ** d_reg_ratio, 0.99 ** d_reg_ratio)
        )

        # weight decay
        def lambda_rule(epoch):
            lr_l = 1.0 - max(0, epoch + self.load_epoch + 1 - 200) / float(200 + 1)
            return lr_l
        self.scheduler_G = torch.optim.lr_scheduler.LambdaLR(self.optimizer_G, lr_lambda=lambda_rule)
        self.scheduler_D = torch.optim.lr_scheduler.LambdaLR(self.optimizer_D, lr_lambda=lambda_rule)

        # load past models
        if self.load_epoch != 0:
            self.load('epoch_' + str(self.load_epoch))

            if get_rank() == 0:
                print('load model: num_epoch {}...'.format(self.load_epoch))

        # set multi-GPUs
        if distributed:
            self.G = torch.nn.parallel.DistributedDataParallel(
                self.G,
                device_ids=[local_rank],
                output_device=local_rank,
                broadcast_buffers=False
            )
            self.D = torch.nn.parallel.DistributedDataParallel(
                self.D,
                device_ids=[local_rank],
                output_device=local_rank,
                broadcast_buffers=False
            )

        if distributed:
            self.G_module = self.G.module
            self.D_module = self.D.module
        else:
            self.G_module = self.G
            self.D_module = self.D

    def requires_grad(self, model, flag=True):
        for p in model.parameters():
            p.requires_grad = flag

    def mixing_noise(self, batch_size, latent_size, prob, device):
        if prob > 0 and random.random() < prob:
            noises = torch.randn(2, batch_size, latent_size, device=device).unbind(0)
            return noises
        else:
            noise = torch.randn(batch_size, latent_size, device=device)  # 16, 512
            return [noise]  # torch.tensor in list, [(torch.tensor), (torch.tensor), ...]

    def g_path_regularize(self, fake_imgs, latents, mean_path_length, decay=0.01):
        noise = torch.randn_like(fake_imgs) / math.sqrt(fake_imgs.shape[2] * fake_imgs.shape[3])
        grad, = autograd.grad(outputs=(fake_imgs * noise).sum(), inputs=latents, create_graph=True)
        path_lengths = torch.sqrt(grad.pow(2).sum(2).mean(1))

        path_mean = mean_path_length + decay * (path_lengths.mean() - mean_path_length)
        path_loss = (path_lengths - path_mean).pow(2).mean()

        return path_loss, path_mean.detach(), path_lengths

    def backward_D_adv(self, real_imgs, batch_size, latent_size, mixing, device):
        # make noise as an input for Generator
        noise = self.mixing_noise(batch_size, latent_size, mixing, device)

        # create fake images
        fake_imgs = self.G(noise)

        # predict real or fake
        fake_pred = self.D(fake_imgs)
        real_pred = self.D(real_imgs)

        # calculate an adversarial loss (logistic loss) / D tries to distinguish real and fake
        real_loss = F.softplus(-real_pred)
        fake_loss = F.softplus(fake_pred)
        d_adv_loss = real_loss.mean() + fake_loss.mean()

        # backward
        d_adv_loss.backward()

        return d_adv_loss

    def backward_D_r1(self, real_imgs, r1, d_reg_every):
        # predict real
        real_imgs.requires_grad = True
        real_pred = self.D(real_imgs)

        # calculate gradient penalty as a r1 loss
        grad_real, = torch.autograd.grad(outputs=real_pred.sum(), inputs=real_imgs, create_graph=True)
        r1_loss = grad_real.pow(2).reshape(grad_real.shape[0], -1).sum(1).mean()

        # backward
        (r1 / 2 * r1_loss * d_reg_every + 0 * real_pred[0]).backward()

        return r1_loss

    def backward_G_adv(self, batch_size, latent_size, mixing, device):
        # make noise as an input for Generator
        noise = self.mixing_noise(batch_size, latent_size, mixing, device)

        # create fake images
        fake_imgs = self.G(noise)

        # predict real or fake
        fake_pred = self.D(fake_imgs)

        # calculate an adversarial loss / G tries to fool D, nonsaturating_loss
        g_adv_loss = F.softplus(-fake_pred).mean()

        # backward
        g_adv_loss.backward()

        return g_adv_loss

    def backward_G_path(self, batch_size, latent_size, mixing, 
                        path_batch_shrink, mean_path_length, path_regularize, g_reg_every, device):
        path_batch_size = max(1, batch_size // path_batch_shrink)
        noise = self.mixing_noise(path_batch_size, latent_size, mixing, device)
        fake_imgs, dlatents = self.G(noise, return_dlatents=True)

        path_loss, mean_path_length, path_lengths = self.g_path_regularize(
            fake_imgs, dlatents, mean_path_length
        )

        g_weighted_path_loss = path_regularize * g_reg_every * path_loss

        if path_batch_shrink:
            g_weighted_path_loss += 0 * fake_imgs[0, 0, 0, 0]

        # backward
        g_weighted_path_loss.backward()

        return path_loss, mean_path_length, path_lengths

    def optimize(self, batch_idx, data):
        real_imgs = data.to(self.device)

        # update Discriminator
        self.optimizer_D.zero_grad()
        d_adv_loss = self.backward_D_adv(
                real_imgs=real_imgs, 
                batch_size=self.batch_size, 
                latent_size=self.latent_size, 
                mixing=self.mixing, 
                device=self.device
                )
        self.optimizer_D.step()

        # apply r1 regularization to Discriminator
        if batch_idx % self.d_reg_every == 0:
            self.optimizer_D.zero_grad()
            self.r1_loss = self.backward_D_r1(
                    real_imgs=real_imgs, 
                    r1=self.r1, 
                    d_reg_every=self.d_reg_every
                    )
            self.optimizer_D.step()
        
        # update Generator
        self.optimizer_G.zero_grad()
        g_adv_loss = self.backward_G_adv(
                batch_size=self.batch_size, 
                latent_size=self.latent_size, 
                mixing=self.mixing, 
                device=self.device
                )
        self.optimizer_G.step()

        # apply path length regularization to Generator
        if batch_idx % self.g_reg_every == 0:
            self.optimizer_G.zero_grad()
            self.path_loss, self.mean_path_length, self.path_lengths = self.backward_G_path(
                    batch_size=self.batch_size, 
                    latent_size=self.latent_size, 
                    mixing=self.mixing,
                    path_batch_shrink=self.path_batch_shrink, 
                    mean_path_length=self.mean_path_length, 
                    path_regularize=self.path_regularize, 
                    g_reg_every=self.g_reg_every,
                    device=self.device
                    )
            self.optimizer_G.step()

            self.mean_path_length_avg = (reduce_sum(self.mean_path_length).item() / get_world_size())
        
        self.loss_dict['d_adv_loss'] = d_adv_loss
        self.loss_dict['r1_loss'] = self.r1_loss
        self.loss_dict['g_adv_loss'] = g_adv_loss
        self.loss_dict['path_loss'] = self.path_loss
        self.loss_dict['path_lengths'] = self.path_lengths.mean()

        # average values among GPUs
        loss_reduced = reduce_loss_dict(self.loss_dict)            

        # average values among Batch
        d_adv_loss_val = loss_reduced["d_adv_loss"].mean().item()
        r1_loss_val = loss_reduced["r1_loss"].mean().item()
        g_adv_loss_val = loss_reduced["g_adv_loss"].mean().item()
        path_loss_val = loss_reduced["path_loss"].mean().item()
        path_lengths_val = loss_reduced["path_lengths"].mean().item() 

        loss_val = [
            d_adv_loss_val,
            r1_loss_val,
            g_adv_loss_val,
            path_loss_val,
            path_lengths_val,
            self.mean_path_length_avg
        ]

        return np.array(loss_val)
        
    def train(self, epoch, data_loader):

        pbar2 = data_loader

        if get_rank() == 0:
            pbar2 = tqdm(
                pbar2,
                dynamic_ncols=True, 
                smoothing=0.01, 
                unit='batch',
                postfix='current epoch: {} / lr: {:.10f}'.format(str(epoch), self.scheduler_G.get_lr()[0]),
                position=0
            )

            running_loss = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0])

        for batch_idx, data in enumerate(pbar2):

            # get losses
            loss_val = self.optimize(batch_idx, data)

            if get_rank() == 0:
                # update pbar2
                pbar2.update(1)

                # set description for pbar2
                pbar2.set_description(
                    (
                        f'd_b: {loss_val[0]:.4f}; ' 
                        f'r1_b: {loss_val[1]:.4f}; '
                        f'g_b: {loss_val[2]:.4f}; ' 
                        f'path_b: {loss_val[3]:.4f}; '
                        f'path_lengths_b: {loss_val[4]:.4f}; '
                        f'mean path_b: {loss_val[5]:.4f}; '
                    )
                )

                # record logs at a batch level       
                wandb.log(
                    {
                        'd_adv_loss_batch': loss_val[0],
                        'r1_loss_batch': loss_val[1],
                        'g_adv_loss_batch': loss_val[2],
                        'path_loss_batch': loss_val[3],
                        'path_lengths_batch': loss_val[4],
                        'mean_path_length_batch': loss_val[5],
                    }
                )

                running_loss += loss_val

        self.scheduler_G.step()
        self.scheduler_D.step()

        if get_rank() == 0:
            running_loss /= len(data_loader)
            return running_loss

    def save_network(self, network, network_label, epoch_label):
        # path to files
        save_filename = '{}_net_{}.pth'.format(network_label, epoch_label)
        save_path = os.path.join(self.log_dir, save_filename)

        # save models
        torch.save(network.state_dict(), save_path)

    def load_network(self, network, network_label, epoch_label):
        # path to files
        load_filename = '{}_net_{}.pth'.format(network_label, epoch_label)
        load_path = os.path.join(self.log_dir, load_filename)

        # load models
        checkpoint = torch.load(load_path, map_location=lambda storage, loc: storage)
        network.load_state_dict(checkpoint)

        del checkpoint  # dereference seems crucial
        torch.cuda.empty_cache()

    def save(self, epoch_label):
        self.save_network(self.G_module, 'Generator', epoch_label)
        self.save_network(self.D_module, 'Discriminator', epoch_label)

    def load(self, epoch_label):
        self.load_network(self.G, 'Generator', epoch_label)
        self.load_network(self.D, 'Discriminator', epoch_label)

    def generate_imgs(self, epoch_label, truncation_target=8, truncation_rate=0.7, truncation_latent=None, return_imgs=False):
        self.G_module.eval()
        imgs = self.G_module([self.sample_z], truncation_target, truncation_rate, truncation_latent)
        img_table_name = 'train_{}.png'.format(epoch_label)
        save_path = os.path.join(self.log_dir, img_table_name)

        self.G_module.train()

        # save and return images
        if return_imgs:
            grid = utils.make_grid(
                imgs, 
                nrow=int(self.n_sample ** 0.5), 
                normalize=True, 
                range=(-1, 1),
            )
            # Add 0.5 after unnormalizing to [0, 255] to round to nearest integer
            ndarr = grid.mul(255).add_(0.5).clamp_(0, 255).permute(1, 2, 0).to('cpu', torch.uint8).numpy()
            im = Image.fromarray(ndarr)
            im.save(save_path)
            return imgs

        # only save images
        else:
            utils.save_image(
                imgs,
                save_path,
                nrow=int(self.n_sample ** 0.5),
                normalize=True,
                range=(-1, 1),
            )



