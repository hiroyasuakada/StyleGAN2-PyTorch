"""
usage:
    base:

"""

import argparse
import os
from pathlib import Path

import torch
from torchvision import utils
from tqdm import tqdm

from model_stylegan2 import Generator


def load_network(network, network_label, epoch_label, path_log):
    # path to files
    load_filename = '{}_net_{}.pth'.format(network_label, epoch_label)
    load_path = os.path.join(path_log, load_filename)

    # load models
    checkpoint = torch.load(load_path, map_location=lambda storage, loc: storage)
    network.load_state_dict(checkpoint)

    del checkpoint  # dereference seems crucial
    torch.cuda.empty_cache()

    return network

def generate_img(device, args, network, mean_latent, epoch_label):
    G_test = network

    if not os.path.exists(Path(args.path_save_dir)):
        os.mkdir(Path(args.path_save_dir))

    with torch.no_grad():
        G_test.eval()
        
        for i in tqdm(range(args.pics)):
            sample_z = torch.randn(args.n_sample, args.latent_size, device=device)
            imgs = G_test(
                [sample_z], 
                truncation_target=args.truncation_target, 
                truncation_rate=args.truncation_rate, 
                truncation_latent=mean_latent
            )

            img_table_name = 'test_{}_{}.png'.format(epoch_label, i)
            save_path = os.path.join(args.path_save_dir, img_table_name)

            utils.save_image(
                imgs,
                save_path,
                nrow=int(args.n_sample ** 0.5),
                normalize=True,
                range=(-1, 1),
            )


if __name__ == '__main__':

    device = 'cuda'

    parser = argparse.ArgumentParser(description='StyleGAN2 generator')

    parser.add_argument(
        '--path_log', type=str, default='checkpoint_2', help='path to the directory of model checkpoints'
    )
    parser.add_argument(
        '--load_epoch', type=int, default=0, help='epochs to generate images'
    )
    parser.add_argument(
        '--path_save_dir', type=str, default='results', help='path to the directory to save images'
    )
    parser.add_argument(
        '--img_size', type=int, default=256, help='image sizes for the model'
    )
    parser.add_argument(
        '--n_sample', type=int, default=16, help ='the number of imgs for each image'
    )
    parser.add_argument(
        "--pics", type=int, default=20, help="the number of images to be generated"
    )
    parser.add_argument(
        '--gpu_ids', nargs='+', type=int, default=[0], help='GPU IDs to be used'
    )
    parser.add_argument(
        '--truncation_target', type=int, default=8, help='the number of layers for applying truncation trick'
    )
    parser.add_argument(
        '--truncation_rate', type=float, default=0.7, help='truncation ratio'
    )
    parser.add_argument(
        '--truncation_latent', 
        type=int, 
        default=4096, 
        help='the number of vectors to calculate mean for truncation trick'
    )
    
    args = parser.parse_args()

    args.latent_size = 512

    # # random seeds
    # torch.manual_seed(1234)
    # np.random.seed(1234)
    # random.seed(1234)

    G_test = Generator(resolution=args.img_size).to(device)
    epoch_label = 'epoch_' + str(args.load_epoch)
    G_test = load_network(G_test, 'Generator', epoch_label, args.path_log)
    G_test = torch.nn.DataParallel(G_test, args.gpu_ids)

    if args.truncation_rate == 1:
        with torch.no_grad():
            mean_latent = G_test.mean_latent(args.truncation_latent, device)
    else:
        mean_latent = None

    # test
    generate_img(device, args, G_test, mean_latent, epoch_label)

