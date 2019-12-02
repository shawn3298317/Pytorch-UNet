import argparse
import logging
import os
import sys

import numpy as np
import torch
import torch.nn as nn
from torch import optim
from torch.utils.data import DataLoader
from tqdm import tqdm

from eval import eval_net
from unet import UNet
from utils import get_ids, split_train_val, get_imgs_and_masks, batch
from utils.modanet_dataset import ModanetDataset
from PIL import Image

def parse_raw_data_into_img_mask():
    train_dataset = ModanetDataset("/projectnb/cs542/shawnlin/image-segmentation/paperdoll/data/chictopia/photos.lmdb",
                                   "/projectnb/cs542/shawnlin/image-segmentation/modanet/annotations/modanet2018_instances_train.json",
                                   64, preprocess=True)
    dev_dataset = ModanetDataset("/projectnb/cs542/shawnlin/image-segmentation/paperdoll/data/chictopia/photos.lmdb",
                                 "/projectnb/cs542/shawnlin/image-segmentation/modanet/annotations/modanet2018_instances_dev.json",
                                 64, preprocess=True)

    train_loader = DataLoader(dataset=train_dataset, batch_size=1, shuffle=False, num_workers=32)
    dev_loader = DataLoader(dataset=dev_dataset, batch_size=1, shuffle=False, num_workers=32)


    for batch_idx, _ in enumerate(tqdm(train_loader)):
        continue
        #img_batch

if __name__ == "__main__":
    parse_raw_data_into_img_mask()

