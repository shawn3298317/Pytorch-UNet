import lmdb
import torch
from torch.utils.data import Dataset
import matplotlib.image as mpimg
from pycocotools.coco import COCO
import io
import numpy as np
import matplotlib.pyplot as plt
import lmdb
from utils.util import annotation_2_multi_binary_mask, annotation_2_multi_mask, reshape_n_crop, get_simple_weight_map, get_unet_weight_map, format_binary_images
import scipy.misc


class ModanetDataset(Dataset):
    """Modanet dataset."""

    def __init__(self, image_lmdb_path, annote_path, input_size, preprocess):
        self.annote_path = annote_path
        self.image_lmdb_path = image_lmdb_path
        #self.training = training
        self.input_size = input_size
        self.preprocess = preprocess

        self.env = lmdb.open(self.image_lmdb_path, map_size=2**36, readonly=True, lock=False)
        self.annotation = COCO(self.annote_path)
        self.id_2_category = {entry["id"]: entry["name"] for entry in self.annotation.loadCats(self.annotation.getCatIds())}

    def __len__(self):
        return len(self.annotation.getImgIds())

    def __getitem__(self, index):

        # Map annotation index to image_id index
        img_id = self.annotation.getImgIds()[index]
        imgs = self.annotation.loadImgs(img_id)
        assert(len(imgs) == 1)
        img_name = imgs[0]["file_name"]

        img = None
        key = str(img_id).encode('ascii')
        with self.env.begin() as t:
            data = t.get(key)
        if not data:
            return [None, None]
        with io.BytesIO(data) as f:
            img = mpimg.imread(f, format='JPG')

        ann_ids = self.annotation.getAnnIds(imgIds=img_id, iscrowd=None)
        annotates = self.annotation.loadAnns(ann_ids)
        multi_mask = annotation_2_multi_mask(img.shape[1], img.shape[0], annotates, self.id_2_category, self.annotation)
        #print("multimask shape:", multi_mask.shape, np.max(multi_mask))
        img = format_binary_images(img, img_name)

        # Training stage random cropping
        crop_img, crop_mask = reshape_n_crop(img, multi_mask, self.input_size)
        #print("crop_mask shape:", crop_mask.shape, np.max(crop_mask))
        #print("crop_img shape:", crop_img.shape)

        if self.preprocess:
            scipy.misc.imsave("./data/imgs/train/%s" % img_name, crop_img)
            scipy.misc.imsave("./data/masks/train/%s.mask.jpg" % img_name.replace(".jpg", ""), crop_mask)
            return []
        else:
            try:
                #print("crop img shape", crop_img.shape)
                #crop_img = np.rollaxis(crop_img, 0, 2).astype('float32')
                crop_img = np.swapaxes(np.swapaxes(crop_img, 0, 2), 1, 2).astype("float32") #np.rollaxis(crop_img, 0, 2).astype('float32')
                #print("after crop img shape", crop_img.shape)
                crop_mask = crop_mask.astype('long')
                #w_map = get_simple_weight_map(crop_mask)
                return [crop_img, crop_mask]
            except np.AxisError:
                print("Axis Error!!", index, img_id, imgs, crop_img.shape)
                return#[None, None]


        #print("crop_img shape:", crop_img.shape)
        #print("crop_mask shape:", crop_mask.shape)

        # TODO: Figure out how to merge multiple patch in Testing stage

        return [crop_img, crop_mask]

