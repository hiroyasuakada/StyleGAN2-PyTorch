import numpy as np
import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F

import time
from tensorboardX import SummaryWriter

from model_networks import Generator, discriminator

# from model_loss import                        
# from dataset import Dataset


###################################################################################################
###                                         StyleGAN2                                           ###
###################################################################################################


class StyleGAN2(object):
    def __init__(self, args, batch_size=1, log_dir='logs', device='cuda:0', lr=0.002, r1=10
                 path_regularize=2, path_batch_shrink=2, g_reg_every=4, d_reg_every=16, mixing=0.9,  
                 mode_train=True):
        
        self.args = args
        self.batch_size = batch_size
        self.log_dir = log_dir
        self.device = device
        print(torch.cuda.is_available())
        self.lr = lr
        self.g_reg_every = g_reg_every
        self.d_reg_every = d_reg_every
        self.latents_dim = 512
        self.r1 = r1
        self.mean_path_length = 0
        self.mean_path_length_avg = 0
        self.mixing = mixing
        self.path_regularize = path_regularize

        if mode_train:
            self.gpu_ids = [0, 1, 2, 3]  # for DLB
        else:
            self.gpu_ids = [0]
        
        self.G = Generator().to(self.device)
        self.D = Discriminator().to(self.device)

        # mmulti-GPUs
        self.G = torch.nn.DataParallel(self.G, self.gpu_ids)
        self.D = torch.nn.DataParallel(self.D, self.gpu_ids)
        
        # loss functions
        self.G_loss = GeneratorLoss(self.device)
        self.D_loss = DiscriminatorLoss(self.device)
        
        # optimize params for G and D
        g_reg_ratio = g_reg_every / (g_reg_every + 1)
        d_reg_ratio = d_reg_every / (d_reg_every + 1)
        self.optimizer_G = torch.optim.Adam(self.G.parameters(), lr=self.lr * g_reg_ratio, 
                                            betas=(0 ** g_reg_ratio, 0.99 ** g_reg_ratio))
        self.optimizer_D = torch.optim.Adam(self.D.parameters(), lr=self.lr * d_reg_ratio, 
                                            betas=(0 ** d_reg_ratio, 0.99 ** d_reg_ratio))

    def mixing_noise(self):
        pass

    def set_input(self, data):
        data_D = data
        data_G = data

        return data_D, data_G

    def backward_D_adv(self, real_imgs):
        # make noise as an input for Generator
        noise = torch.randn(self.batch, self.latents_dim, device=self.device)

        # create fake images
        fake_imgs, _ = self.G(noise)

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

    def backward_D_r1(self, real_imgs):
        # predict real
        real_imgs.requires_grad = True
        real_pred = self.D(real_imgs)

        # calculate gradient penalty as a r1 loss
        grad_real, = torch.autograd.grad(outputs=real_pred.sum(), inputs=real_imgs, create_graph=True)
        d_r1_loss = grad_real.pow(2).reshape(grad_real.shape[0], -1).sum(1).mean()

        # backward
        (self.r1 / 2 * d_r1_loss * self.d_reg_every + 0 * real_pred[0]).backward()

        return d_r1_loss

    def backward_G_adv(self, data):
        # make noise as an input for Generator
        noise = self.mixing_noise(self.batch_size, self.latents_dim, self.mixing, device=self.device)
        torch.randn(self.batch, self.latents_dim, device=self.device)
        
        # create fake images
        fake_imgs, _ = self.G(noise)

        # predict real or fake
        fake_pred = self.D(fake_imgs)

        # calculate an adversarial loss / G tries to fool D 
        g_adv_loss = F.softplus(-fake_pred).mean()

        # backward
        g_adv_loss.backward()

        return g_adv_loss

    def backward_G_path(self, data):
        path_batch_size = max(1, self.batch_size // self.path_batch_shrink)
        noise = self.mixing_noise(path_batch_size, self.latents_dim, self.mixing, device=self.device)
        fake_imgs, latents = self.G(noise, return_latents=True)

        noise = torch.randn_like(fake_imgs) / math.sqrt(fake_imgs.shape[2] * fake_imgs.shape[3])
        grad, = torch.autograd.grad(outputs=(fake__imgs * noise).sum(), inputs=latents, crate_graph=True)

        path_lengths = torch.sqrt(grad.pow(2).sum(2).mean(1))
        path_mean = mean_path_length + 0.01 * (path_lengths.mean() - mean_path_length)
        path_loss = (path_lengths - path_mean).pow(2).mean()

        mean_path_length = path_mean.detach()

        g_weighted_path_loss = self.path_regularize * self.g_reg_every * path_loss

        if self.path_batch_shrink:
            g_weighted_path_loss += 0 * fake_imgs[0, 0, 0, 0]

        # backward
        g_weighted_path_loss.backward()

        return g_weighted_path_loss

    def optimize(self, batch_idx, data):
        data_D, data_G = self.set_input(data)

        # update Discriminator
        self.optimizer_D.zero_grad()
        d_adv_loss = self.backward_D_adv(data)
        self.optimizer_D.step

        # apply r1 regularization to Discriminator
        if batch_idx % self.d_reg_every == 0:
            self.optimizer_D.zero_grad()
            d_r1_loss = self.backward_D_r1(data)
            self.optimizer_D.step
        else:
            d_r1_loss = None
        
        # update Generator
        self.optimizer_G.zero_grad()
        g_adv_loss = self.backward_G_adv(self.data)
        self.optimizer_G.step

        if batch % self.g_reg_every == 0:
            self.optimizer_G.zero_grad()
            g_weighted_path_loss = self.backward_G_path(self.data)
            self.optimizer_G.step
        else:
            g_weighted_path_loss = None

        # self.mean_path_length_avg = (
        #     reduce_sum(mean_path_length).item() / get_world_size()
        # )


        losses = [d_adv_loss, d_r1_loss, 
                  g_adv_loss, g_weighted_path_loss]

        return np.array(losses).astype(np.float32)
        
    def train(self, data_loader):
        running_loss = np.array([0.0, 0.0, 0.0, 0.0])
        time_list = []

        for batch_idx, data in enumerate(data_loader):
            # count time 1
            t1 = time.perf_counter()

            # get losses
            losses = self.optimize(batch_idx, data)
            running_loss += losses

            # count time 2 
            t2 = time.perf_counter()
            get_processing_time = t2 - t1
            time_list.append(get_processing_time)

            # print batch and processing time, when idx == 500 
            if batch_idx % 500 == 0:
                print('batch: {} / elapsed_time: {} sec'.format(batch_idx, sum(time_list)))
                time_list = []

        running_loss /= len(data_loader)
        return running_loss

    def save_network(network, network_label, epoch_label):
        # path to files
        save_filename = '{}_net_{}_epoch.pth'.format(network_label, epoch_label)
        save_path = os.path.join(self.log_dir, save_filename)

        # save models on CPU
        torch.save(network.cpu().state_dict(), save_path)

        # return models to GPU
        network.to(self.device)

    def load_network(network, network_label, epoch_label):
        # path to files
        load_filename = '{}_net_{}_epoch.pth'.format(network_label, epoch_label)
        load_path = os.path.join(self.log_dir, load_filename)

        # load models
        network.load_state_dict(torch.load(load_path))

    def save(self, epoch_label):
        self.save_network(self.G, 'Generator', epoch_label)
        self.save_network(self.D, 'Discriminator', epoch_label)

    def load(self, epoch_label):
        self.load_network(self.G, 'Generator', epoch_label)
        self.load_network(self.D, 'Discriminator', epoch_label)

