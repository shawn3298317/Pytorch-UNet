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
from tensorboardX import SummaryWriter

from eval import eval_net
from unet import UNet
from utils.modanet_dataset import ModanetDataset

dir_checkpoint = "./ckpt/"

def train_net(net,
              device,
              epochs=5,
              batch_size=1,
              lr=0.001,
              save_cp=True,
              img_scale=0.5,
              writer=None):

    train_dataset = ModanetDataset("/projectnb/cs542/shawnlin/image-segmentation/paperdoll/data/chictopia/photos.lmdb",
                                   "/projectnb/cs542/shawnlin/image-segmentation/modanet/annotations/modanet2018_instances_train.json",
                                   400, False)
    dev_dataset = ModanetDataset("/projectnb/cs542/shawnlin/image-segmentation/paperdoll/data/chictopia/photos.lmdb",
                                 "/projectnb/cs542/shawnlin/image-segmentation/modanet/annotations/modanet2018_instances_dev.json",
                                 400, False)


    n_train = len(train_dataset)
    n_val = len(dev_dataset)

    logging.info(f'''Starting training:
        Epochs:          {epochs}
        Batch size:      {batch_size}
        Learning rate:   {lr}
        Training size:   {n_train}
        Validation size: {n_val}
        Checkpoints:     {save_cp}
        Device:          {device.type}
        Images scaling:  {img_scale}
    ''')


    train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True, num_workers=8)
    dev_loader = DataLoader(dataset=dev_dataset, batch_size=1, shuffle=True, num_workers=4)

    optimizer = optim.Adam(net.parameters(), lr=lr)
    if net.n_classes > 1:
        criterion = nn.CrossEntropyLoss()
    else:
        criterion = nn.BCEWithLogitsLoss()

    """
    img_enc = None
    if use_pretrain_cnn == "resnet":
        img_enc = models.resnet18(pretrained=True)
        modules = list(img_enc.children())[:-2]
        img_enc = nn.Sequential(*modules)
        for params in img_enc.parameters():
            params.requires_grad = False
            img_enc = img_enc.cuda()
        img_enc.eval()
    """
    for epoch in range(epochs):
        net.train()

        epoch_loss = 0

        for batch_idx, batch in enumerate(tqdm(train_loader)):
            #if batch_idx > 50:
            #    break

            if batch is None:
                continue

            img_batch, mask_batch = batch
            img_batch = img_batch.to(device)
            mask_batch = mask_batch.to(device)


            optimizer.zero_grad()
            masks_pred = net(img_batch)

            #print("mask shape", mask_batch.shape)
            #print("mask_pred shape", masks_pred.shape)
            loss = criterion(masks_pred, mask_batch)
            epoch_loss += loss.item()

            loss.backward()
            optimizer.step()

            if writer is not None:
                train_step = epoch*len(train_loader) + (batch_idx + 1)
                writer.add_scalar('train/loss', loss.data, train_step)
                if train_step % 10 == 0:
                    writer.flush()

            print("Train Epoch: %i, batch: %i, Loss: %.6f" % (epoch, batch_idx, loss.data))

        if save_cp:
            try:
                os.mkdir(dir_checkpoint)
                logging.info('Created checkpoint directory')
            except OSError:
                pass
            torch.save(net.state_dict(), dir_checkpoint + f'CP_epoch{epoch + 1}.pth')
            logging.info(f'Checkpoint {epoch + 1} saved !')

        val_score = eval_net(net, dev_loader, device)
        if writer is not None:
            writer.add_scalar("val/loss", val_score, epoch)

        if net.n_classes > 1:
            logging.info('Validation cross entropy: {}'.format(val_score))

        else:
            logging.info('Validation Dice Coeff: {}'.format(val_score))


def get_args():
    parser = argparse.ArgumentParser(description='Train the UNet on images and target masks',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('-e', '--epochs', metavar='E', type=int, default=10,
                        help='Number of epochs', dest='epochs')
    parser.add_argument('-b', '--batch-size', metavar='B', type=int, nargs='?', default=4,
                        help='Batch size', dest='batchsize')
    parser.add_argument('-l', '--learning-rate', metavar='LR', type=float, nargs='?', default=0.001,
                        help='Learning rate', dest='lr')
    parser.add_argument('-f', '--load', dest='load', type=str, default=False,
                        help='Load model from a .pth file')
    parser.add_argument('-s', '--scale', dest='scale', type=float, default=0.5,
                        help='Downscaling factor of the images')
    parser.add_argument('-v', '--validation', dest='val', type=float, default=15.0,
                        help='Percent of the data that is used as validation (0-100)')

    return parser.parse_args()


def pretrain_checks():
    pass
    #if len(imgs) != len(masks):
    #    logging.warning(f'The number of images and masks do not match ! '
    #                    f'{len(imgs)} images and {len(masks)} masks detected in the data folder.')


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
    args = get_args()
    pretrain_checks()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logging.info(f'Using device {device}')

    # Change here to adapt to your data
    # n_channels=3 for RGB images
    # n_classes is the number of probabilities you want to get per pixel
    #   - For 1 class and background, use n_classes=1
    #   - For 2 classes, use n_classes=1
    #   - For N > 2 classes, use n_classes=N
    net = UNet(n_channels=3, n_classes=14)
    logging.info(f'Network:\n'
                 f'\t{net.n_channels} input channels\n'
                 f'\t{net.n_classes} output channels (classes)\n'
                 f'\t{"Bilinear" if net.bilinear else "Dilated conv"} upscaling')

    if args.load:
        net.load_state_dict(torch.load(args.load, map_location=device))
        logging.info(f'Model loaded from {args.load}')

    net.to(device=device)
    # faster convolutions, but more memory
    # cudnn.benchmark = True

    try:
        writer = SummaryWriter()
        train_net(net=net,
                  epochs=args.epochs,
                  batch_size=args.batchsize,
                  lr=args.lr,
                  device=device,
                  img_scale=args.scale,
                  writer=writer)
    except KeyboardInterrupt:
        torch.save(net.state_dict(), 'INTERRUPTED.pth')
        logging.info('Saved interrupt')
        try:
            sys.exit(0)
        except SystemExit:
            os._exit(0)
