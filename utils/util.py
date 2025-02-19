import random
import numpy as np
from skimage.transform import resize
import cv2

def hwc_to_chw(img):
    return np.transpose(img, axes=[2, 0, 1])


def resize_and_crop(pilimg, scale=0.5, final_height=None):
    w = pilimg.size[0]
    h = pilimg.size[1]
    newW = int(w * scale)
    newH = int(h * scale)

    if not final_height:
        diff = 0
    else:
        diff = newH - final_height

    img = pilimg.resize((newW, newH))
    img = img.crop((0, diff // 2, newW, newH - diff // 2))
    ar = np.array(img, dtype=np.float32)
    if len(ar.shape) == 2:
        # for greyscale images, add a new axis
        ar = np.expand_dims(ar, axis=2)
    return ar

def batch(iterable, batch_size):
    """Yields lists by batch"""
    b = []
    for i, t in enumerate(iterable):
        b.append(t)
        if (i + 1) % batch_size == 0:
            yield b
            b = []

    if len(b) > 0:
        yield b


def split_train_val(dataset, val_percent=0.05):
    dataset = list(dataset)
    length = len(dataset)
    n = int(length * val_percent)
    random.shuffle(dataset)
    return {'train': dataset[:-n], 'val': dataset[-n:]}


def normalize(x):
    return x / 255


# credits to https://stackoverflow.com/users/6076729/manuel-lagunas
def rle_encode(mask_image):
    pixels = mask_image.flatten()
    # We avoid issues with '1' at the start or end (at the corners of
    # the original image) by setting those pixels to '0' explicitly.
    # We do not expect these to be non-zero for an accurate mask,
    # so this should not harm the score.
    pixels[0] = 0
    pixels[-1] = 0
    runs = np.where(pixels[1:] != pixels[:-1])[0] + 2
    runs[1::2] = runs[1::2] - runs[:-1:2]
    return runs

def annotation_2_multi_binary_mask(OUT_WIDTH, OUT_HEIGHT, annotates, id_2_category, annotation):
    multi_binary_mask = np.zeros((OUT_HEIGHT, OUT_WIDTH, len(id_2_category)+1))
    for ann in annotates:
        mask = annotation.annToMask(ann)
        multi_binary_mask[:, :, ann["category_id"]] = mask
    return multi_binary_mask

def annotation_2_multi_mask(OUT_WIDTH, OUT_HEIGHT, annotates, id_2_category, annotation):
    multi_binary_mask = np.zeros((OUT_HEIGHT, OUT_WIDTH, len(id_2_category)+1))
    for ann in annotates:
        mask = annotation.annToMask(ann)
        multi_binary_mask[:, :, ann["category_id"]] = mask

    multi_mask = np.argmax(multi_binary_mask, axis=2)
    return multi_mask

def format_binary_images(img, img_name):
    if len(img.shape) == 2:
        print("Formatting image: %s!" % img_name)
        img_fix = np.zeros((img.shape[0], img.shape[1], 3))
        img_fix[:, :, 0] = img
        return img_fix
    elif len(img.shape) == 3:
        return img

def reshape_n_crop(img, mask, size):
    scaled_height = int(size/img.shape[1]*img.shape[0])
    #print("Reshaping:", img.shape, mask.shape, np.max(mask), scaled_height, np.max(img))
    img_resize = cv2.resize(img, (size, scaled_height))
    mask_resize = cv2.resize(mask, (size, scaled_height), interpolation=cv2.INTER_NEAREST)
    #print("After Reshaping:", img_resize.shape, mask_resize.shape, np.max(mask_resize))

    crop_x = np.random.randint(0, (img_resize.shape[0] - img_resize.shape[1]))
    img_crop = img_resize[crop_x:(crop_x+size), :]
    mask_crop = mask_resize[crop_x:(crop_x+size), :]

    #return img, mask
    return img_crop, mask_crop

def get_simple_weight_map(mask):
    w_map = np.zeros((mask.shape[0], mask.shape[1]))
    w_map[mask > 0] = 5.0
    return w_map

def get_unet_weight_map(mask):
    return None
