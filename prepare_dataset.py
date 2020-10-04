"""
-------------------------------------------------
   File Name:    prepare_dataset.py
   Author:       Hiroyasu Akada
   Date:         1, Oct, 2020
   Description:  Modified from 
                 https://github.com/rosinality/stylegan2-pytorch/blob/master/prepare_data.py

    Usage:       python prepare_data.py --out LMDB_PATH --n_worker N_WORKER --size SIZE1,SIZE2,SIZE3,... DATASET_PATH
        For Author:
        python3 prepare_dataset.py --out lmdb_256_70000 --size 256 --n_worker 16 ffhq_r1024
-------------------------------------------------
"""

import argparse
from io import BytesIO
import multiprocessing
from functools import partial

from PIL import Image
import lmdb
from tqdm import tqdm
from torchvision import datasets
from torchvision.transforms import functional as trans_fn


def resize_and_convert(img, size, resample, quality=100):
    img = trans_fn.resize(img, size, resample)
    img = trans_fn.center_crop(img, size)
    buffer = BytesIO()
    img.save(buffer, format="jpeg", quality=quality)
    val = buffer.getvalue()

    return val


def resize_multiple(img, size=256, resample=Image.LANCZOS, quality=100):
    imgs = []
    imgs.append(resize_and_convert(img, size, resample, quality))

    return imgs


def resize_worker(img_file, size, resample):
    i, file = img_file
    img = Image.open(file)
    img = img.convert("RGB")
    out = resize_multiple(img, size=size, resample=resample)

    return i, out


def prepare(env, dataset, n_worker, size=256, resample=Image.LANCZOS):
    resize_fn = partial(resize_worker, size=size, resample=resample)

    files = sorted(dataset.imgs, key=lambda x: x[0])
    files = [(i, file) for i, (file, label) in enumerate(files)]
    total = 0

    with multiprocessing.Pool(n_worker) as pool:
        for i, imgs in tqdm(pool.imap_unordered(resize_fn, files)):
            for img in imgs:
                key = f"{size}-{str(i).zfill(5)}".encode("utf-8")

                with env.begin(write=True) as txn:
                    txn.put(key, img)

            total += 1

        with env.begin(write=True) as txn:
            txn.put("length".encode("utf-8"), str(total).encode("utf-8"))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Crate LMDB Dataset for Images")
    parser.add_argument("--out", type=str, help="filename of the result lmdb dataset")
    parser.add_argument(
        "--size",
        type=int,
        default=256,
        help="resolution of images for the dataset",
    )
    parser.add_argument(
        "--n_worker",
        type=int,
        default=4,
        help="number of workers for preparing dataset",
    )
    parser.add_argument(
        "--resample",
        type=str,
        default="lanczos",
        help="resampling methods for resizing images",
    )
    parser.add_argument("path", type=str, help="path to the image dataset")

    args = parser.parse_args()

    resample_map = {"lanczos": Image.LANCZOS, "bilinear": Image.BILINEAR}
    resample = resample_map[args.resample]

    print("Make dataset of image sizes: {}".format(args.size))

    imgset = datasets.ImageFolder(args.path)

    with lmdb.open(args.out, map_size=1024 ** 4, readahead=False) as env:
        prepare(env, imgset, args.n_worker, size=args.size, resample=resample)