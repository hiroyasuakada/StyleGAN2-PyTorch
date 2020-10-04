import argparse
from pathlib import Path
import pickle
import PIL.Image

import numpy as np
import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F
import tensorflow as tf

import dnnlib

from tempfile import TemporaryDirectory

from model_stylegan2 import Generator
from weights_conversion import WeightsConverter


# initialize tensorflow
def init_tf():
    tf_random_seed = np.random.randint(1 << 31)
    tf.set_random_seed(tf_random_seed)

    config_proto = tf.ConfigProto()
    config_proto.graph_options.place_pruned_graph = True
    config_proto.gpu_options.allow_growth = True

    session = tf.Session(config=config_proto)
    session._default_session = session.as_default()
    session._default_session.enforce_nesting = False
    session._default_session.__enter__()

    return session


# to dnnlib.submit_run
def convert_from_tf_to_np(args, src): 
    init_tf()

    # load the officail pre-trained weights of Tensorflow
    with (Path(args.weight_dir)/src['src_weight']).open('rb') as f:
        *_, Gs = pickle.load(f)

    # save the weights as a numpy format
    print('src_ndarray_weight save...')
    ndarrays = {k:v.eval() for k,v in Gs.vars.items()}
    [print(k,v.shape) for k,v in ndarrays.items()]
    with (Path(args.weight_dir)/src['src_ndarray_weight']).open('wb') as f:
        pickle.dump(ndarrays,f)


if __name__ == '__main__':

    # command line
    parser = argparse.ArgumentParser(
        description='Convert pre-trained weights of the official Tensorflow model to the ones of Pytorch ')
    parser.add_argument('-w','--weight_dir', 
                        type=str, 
                        default='original_implementation_by_tf', 
                        help='dict where pre-trained weights are saved')
    parser.add_argument('-o','--output_dir', 
                        type=str, 
                        default='checkpoint', 
                        help='dict where generated images will be saved')
    parser.add_argument('-rn', '--rename', 
                        type=str,
                        default='stylegan2_pytorch_state_dict.pth',
                        help='file name of converted weights')
    args = parser.parse_args()

    src = {
        'src_weight'  : 'stylegan2-ffhq-config-f.pkl',
        'src_ndarray_weight' : 'stylegan2_ndarray.pkl',
    }

    # save the official weights of Tensorflow as a numpy format
    with TemporaryDirectory() as dir_name:
        config = dnnlib.SubmitConfig()
        config.local.do_not_copy_source_files   = True
        config.run_dir_root                     = dir_name
        dnnlib.submit_run(config, 'convert_weights.convert_from_tf_to_np', args=args, src=src)

    # build model
    print('\n build models... \n')
    generator = Generator()
    base_state_dict = generator.state_dict()

    # load the pre-trained weights of Numpy-format
    print('\n load pre-trained weights of numpy format... \n')
    with (Path(args.weight_dir)/src['src_ndarray_weight']).open('rb') as f:
        src_dict_tf = pickle.load(f)
    
    # translate the pre-trained weights into Pytorch version
    print('\n convert weights... \n')
    WC = WeightsConverter()
    new_dict_pt = WC.convert(src_dict_tf)
    for i in new_dict_pt:
        base_state_dict[i] = new_dict_pt[i]

    print('\n try to set state_dict... \n')
    generator.load_state_dict(base_state_dict)

    print('\n try to set state_dict... ok \n')   

    print('\n weight save... \n')
    torch.save(generator.state_dict(), str(Path(args.output_dir + '/' + args.rename)))

    print('\n all done \n')