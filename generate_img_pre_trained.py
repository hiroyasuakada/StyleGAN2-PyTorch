import argparse
from pathlib import Path
import pickle
import os

import numpy as np
import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F

from model_stylegan2 import Generator


# make table for generated imgs
def make_table_for_imgs(imgs, num_H, num_W, resolution):
    H = W = resolution
    num_imgs = num_H * num_W
    table = np.zeros((H * num_H, W * num_W, 3),dtype=np.uint8)

    for i,p in enumerate(imgs[:num_imgs]):
        h, w = i // num_W, i % num_W
        table[H * h:H * -~h, W * w:W * -~w, :] = p[:, :, ::-1]

    return table


if __name__ == '__main__':

    # gpu or cpu
    if torch.cuda.is_available():
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')

    parser = argparse.ArgumentParser(description='Generate images with pre-trained weights of StyleGAN2')
    parser.add_argument(
        '--path_weights', 
        type=str, 
        default='checkpoint_pre_trained/stylegan2_pytorch_state_dict.pth', 
        help='path to pre-trained weights'
    )
    parser.add_argument(
        '--output_dir', 
        type=str, 
        default='results_pre_trained', 
        help='path to dict where generated images will be saved'
    )
    parser.add_argument(
        '--imgs_name', 
        type=str, 
        default='generated_imgs_pre_trained.png', 
        help='name of table of generated images'
    )
    parser.add_argument(
        '--resolution', 
        type=int, 
        default=1024,
        help='resolution of individual images'
    )
    parser.add_argument(
        '--num_rows',
        type=int,
        default=4,
        help='the number of rows of image table'
    )
    parser.add_argument(
        '--num_columns',
        type=int,
        default=4,
        help='the number of columns of image table'
    )
    parser.add_argument(
        '--batch_size', 
        type=int, 
        default=1,
        help='batch size to be passed through generator'
    )
    parser.add_argument(
        '--path_latents',
        type=str,
        default=None,
        help='path to latents which you want to use to generate images'
    )
    args = parser.parse_args()

    if not os.path.exists(Path(args.output_dir)):
        os.mkdir(Path(args.output_dir))

    ## build a model
    print('\n build models... \n')
    generator = Generator()
    base_state_dict = generator.state_dict()

    # load pre-trained weights of Pytorch
    print('\n set state_dict... \n')
    new_dict_pt = torch.load(Path(args.path_weights))
    generator.load_state_dict(new_dict_pt)
    
    # configs for images and table
    num_H = args.num_rows
    num_W = args.num_columns
    N = num_images = num_H * num_W
    resolution = args.resolution

    if args.path_latents is None:

        # create new latents
        print('\n create latents... \n')
        latents = np.random.RandomState(5).randn(N, 512)
        latents = torch.from_numpy(latents.astype(np.float32))
        
        # save latents for later use
        with (Path(args.output_dir)/'used_latents.pkl').open('wb') as f:
            pickle.dump(latents, f)

    else:
        # load latents
        print('\n load latents... \n')
        with (Path(args.output_dir)/'latents2.pkl').open('rb') as f:
            latents = pickle.load(f)
        latents = torch.from_numpy(latents.astype(np.float32))
    
    print('\n network forward... \n')
    with torch.no_grad():
        N,_ = latents.shape
        generator.to(device)
        imgs = np.empty((N, args.resolution, args.resolution, 3), dtype=np.uint8)

        for i in range(0, N, args.batch_size):
            j = min(i + args.batch_size, N)
            z = latents[i:j].to(device)
            img = generator([z])
            normalized = (img.clamp(-1, 1) + 1) / 2 * 255
            imgs[i:j] = normalized.permute(0, 2, 3, 1).cpu().numpy().astype(np.uint8)
            del z, img, normalized
 
    print('\n images output... \n')
    cv2.imwrite(str(Path(args.output_dir)/args.imgs_name), make_table_for_imgs(imgs, num_H, num_W, resolution))
    
    print('\n all done \n')