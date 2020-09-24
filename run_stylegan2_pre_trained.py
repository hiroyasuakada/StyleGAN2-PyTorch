import argparse
from pathlib import Path
import pickle

import numpy as np
import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F

from network_stylegan2 import Generator
from weights_conversion import WeightsConverter

if __name__ == '__main__':

    # gpu or cpu
    if torch.cuda.is_available():
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')

    # command line
    parser = argparse.ArgumentParser(description='Run StyleGAN2 with pre-trained weights by Pytorch')
    parser.add_argument('--weight_dir', type=str, default='original_implementation_by_tf', help='dict where pre-trained weights')
    parser.add_argument('--output_dir', type=str, default='generated_imgs', help='dict where generated images will be saved')
    parser.add_argument('--batch_size', type=int, default=1)
    parser.add_argument('--resolution', type=int, default=1024)
    args = parser.parse_args()

    ## using pre-trained model of the original StyleGAN2  
    # build model
    generator = Generator()
    base_state_dict = generator.state_dict()

    # config for pre-trained weights
    cfg = {
        'src_weight': 'stylegan2_ndarray.pkl',
        'src_latent': 'latents2.pkl',
        'dst_image' : 'stylegan2_pt.png',
        'dst_weight': 'stylegan2_state_dict.pth'
    }

    # load pre-trained weights 
    print('\n load pre-trained weights... \n')
    with (Path(args.weight_dir)/cfg['src_weight']).open('rb') as f:
        src_dict_tf = pickle.load(f)
    
    # translate the pre-trained weights from Tensorflow into Pytorch
    print('\n set state_dict... \n')
    WC = WeightsConverter()
    new_dict_pt = WC.convert(src_dict_tf)
    generator.load_state_dict(new_dict_pt)

    # load latents
    print('\n load latents... \n')
    with (Path(args.output_dir)/cfg['src_latent']).open('rb') as f:
        latents = pickle.load(f)
    latents = torch.from_numpy(latents.astype(np.float32))
    
    print('\n network forward... \n')
    with torch.no_grad():
        N,_ = latents.shape
        generator.to(device)
        imgs = np.empty((N,args.resolution,args.resolution,3), dtype=np.uint8)

        for i in range(0, N, args.batch_size):
            j = min(i + args.batch_size, N)
            z = latents[i:j].to(device)
            img = generator(z)
            normalized = (img.clamp(-1, 1) + 1) / 2 * 255
            imgs[i:j] = normalized.permute(0, 2, 3, 1).cpu().numpy().astype(np.uint8)
            del z, img, normalized

    # make table for generated imgs
    def make_table_for_imgs(imgs):
        # the number of imgs
        num_H, num_W = 4,4

        # resolution
        H = W = args.resolution

        num_imgs = num_H * num_W
        table = np.zeros((H * num_H, W * num_W, 3),dtype=np.uint8)

        for i,p in enumerate(imgs[:num_imgs]):
            h, w = i // num_W, i % num_W
            table[H * h:H * -~h, W * w:W * -~w, :] = p[:, :, ::-1]

        return table
 
    print('\n images output... \n')
    cv2.imwrite(str(Path(args.output_dir)/cfg['dst_image']), make_table_for_imgs(imgs))
    
    print('\n weight save... \n')
    torch.save(generator.state_dict(), str(Path(args.weight_dir)/cfg['dst_weight']))
    
    print('\n all done \n')