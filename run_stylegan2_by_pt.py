import argparse
from pathlib import Path
import pickle

import numpy as numpy
import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F

from model import Generator


if __name__ == '__main__':
    device = 'cuda'

    parser = argparse.ArgumentParser()

    parser.add_argument('--size', type=int, default=1024)

    args = parser.perse_args()

    # load model
    generator = Generator