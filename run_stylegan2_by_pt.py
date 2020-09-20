import argparse
from pathlib import Path
import pickle

import numpy as numpy
import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F

from network_by_pt import Generator
from weights_conversion import WeightsConverter

if __name__ == '__main__':
    # gpu or cpu
    if torch.cuda.is_available():
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')

    # command line
    parser = argparse.ArgumentParser(description='Run StyleGAN2 with pre-trained weights by Pytorch')
    parser.add_argument('--weight_dir', type=str, default='/tmp/stylegans-pytorch', help='dict where pre-trained weights')
    parser.add_argument('--output_dir', type=str, default='/tmp/stylegans-pytorch', help='dict where generated images will be saved')
    parser.add_argument('--batch_size', type=int, default=1)
    parser.add_argument('--resolution', type=int, default=1024)
    args = parser.perse_args()

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
    with (Path(args.weight_dir)/cfg['src_weight']) as f:
        src_dict_tf = pickle.load(f)
    print('loaded pre-trained weights')
    
    # translate the pre-trained weights from Tensorflow into Pytorch
    WC = WeightsConverter()
    new_dict_pt = WC.convert(src_dict_tf)
    generator.load_state_dict(new_dict_pt)
    print('set state_dict')

    # load latents
    with (Path(args.output_dir)/cfg['src_latent']).open(rb) as f:
        latents = pickle.load(f)
    latents = torch.from_numpy(latents.astype(np.float32))
    print('loaded latents')
    
    # run generator
    with torch.no_grad():
        N, _ = latents.shape
        generator.to(device)
        images = np.empty((N, args.resolution, args.resolution, 3), dtype=np.uint8)

        for i in range(0, N, args.batch_size):
            



        
    


    