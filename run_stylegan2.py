import argparse
from pathlib import Path
import pickle

import numpy as numpy
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
    parser.add_argument('--weight_dir', type=str, default='/original_implementation', help='dict where pre-trained weights')
    parser.add_argument('--output_dir', type=str, default='/generated_imgs', help='dict where generated images will be saved')
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
    
    print('network forward...')
    device = torch.device('cuda:0') if torch.cuda.is_available() and args.device=='gpu' else torch.device('cpu')
    with torch.no_grad():
        N,_ = latents.shape
        generator.to(device)
        images = np.empty((N,args.resolution,args.resolution,3),dtype=np.uint8)

        for i in range(0,N,args.batch_size):
            j = min(i+args.batch_size,N)
            z = latents[i:j].to(device)
            img = generator(z)
            normalized = (img.clamp(-1,1)+1)/2*255
            images[i:j] = normalized.permute(0,2,3,1).cpu().numpy().astype(np.uint8)
            del z, img, normalized

    # 出力を並べる関数
    def make_table(imgs):
        # 出力する個数，解像度
        num_H, num_W = 4,4
        H = W = args.resolution
        num_images = num_H*num_W

        canvas = np.zeros((H*num_H,W*num_W,3),dtype=np.uint8)
        for i,p in enumerate(imgs[:num_images]):
            h,w = i//num_W, i%num_W
            canvas[H*h:H*-~h,W*w:W*-~w,:] = p[:,:,::-1]
        return canvas
 
    print('image output...')
    cv2.imwrite(str(Path(args.output_dir)/cfg['dst_image']), make_table(images))
    
    print('weight save...')
    torch.save(generator.state_dict(),str(Path(args.weight_dir)/cfg['dst_weight']))
    
    print('all done')