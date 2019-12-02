import torch
import torch.nn.functional as F
from tqdm import tqdm
import scipy.misc
import numpy as np

from dice_loss import dice_coeff


def eval_net(net, dev_loader, device):
    """Evaluation without the densecrf with the dice coefficient"""
    net.eval()
    tot = 0

    # use no grad to tell model not to perform back prop calc optimizations in forward pass to save time
    with torch.no_grad():
        for batch_idx, (img_batch, mask_batch) in enumerate(dev_loader):
            img_batch = img_batch.to(device)
            mask_batch = mask_batch.to(device)

            mask_pred = net(img_batch)
            #print("mask_pred shape", mask_pred.shape)
            #mask_pred =
            #mask_pred = (mask_pred > 0.5).float()

            if batch_idx % 249 == 1:
                img = img_batch.data.cpu().numpy()[0]
                img = np.swapaxes(img, 0, 2)
                pred = mask_pred[0].data.cpu().numpy()[0]
                print("imgshape", img.shape)
                print("predshape", pred.shape)
                scipy.misc.imsave('eval_%i.img.jpg' % batch_idx, img)
                scipy.misc.imsave('eval_%i.predict.jpg' % batch_idx, pred)

            if net.n_classes > 1:
                tot += F.cross_entropy(mask_pred, mask_batch).item()
            else:
                tot += dice_coeff(mask_pred, mask_batch.squeeze(dim=1)).item()

    return tot / len(dev_loader)

def intersectionAndUnion(imPred, imLab, numClass):
    imPred = np.asarray(imPred).copy()
    imLab = np.asarray(imLab).copy()

    imPred += 1
    imLab += 1
    # Remove classes from unlabeled pixels in gt image.
    # We should not penalize detections in unlabeled portions of the image.
    imPred = imPred * (imLab > 0)

    # Compute area intersection:
    intersection = imPred * (imPred == imLab)
    (area_intersection, _) = np.histogram(
        intersection, bins=numClass, range=(1, numClass))

    # Compute area union:
    (area_pred, _) = np.histogram(imPred, bins=numClass, range=(1, numClass))
    (area_lab, _) = np.histogram(imLab, bins=numClass, range=(1, numClass))
    area_union = area_pred + area_lab - area_intersection

    return (area_intersection, area_union)

