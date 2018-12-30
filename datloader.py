from __future__ import  absolute_import
from __future__ import  division
import torch 
from vocdata import BboxDat
from skimage import transform as sktsf
from torchvision import transforms 
import numpy as np
import random

voc_data_dir = 'C:/Users/Dhaya/Documents/DL_Project/VOCdevkit/VOC2012/'
min_size = 600
max_size = 1000

def preprocess(image, min_size=600, max_size=1000):
    """
    Preprocess an image for feature extraction.
    """
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
    	std=[0.229, 0.224, 0.225])


    C, H, W = image.shape
    scale1 = min_size / min(H, W)
    scale2 = max_size / max(H, W)
    scale = min(scale1, scale2)
    image = image / 255.
    image = sktsf.resize(image, (C, H * scale, W * scale), mode='reflect',anti_aliasing=False)	

    image = normalize(torch.from_numpy(image))
    return image.numpy()


def resize_bbox(bbox, in_size, out_size):

    """ Resize bounding boxes according to image resize """

    bbox = bbox.copy()
    y_scale = float(out_size[0]) / in_size[0]
    x_scale = float(out_size[1]) / in_size[1]
    bbox[:, 0] = y_scale * bbox[:, 0]
    bbox[:, 2] = y_scale * bbox[:, 2]
    bbox[:, 1] = x_scale * bbox[:, 1]
    bbox[:, 3] = x_scale * bbox[:, 3]
    return bbox


def flip_bbox(bbox, size, flip=False):

    """ Flip bounding boxes accordingly """

    H, W = size
    bbox = bbox.copy()

    if flip: #this is just doing symmetric flipping of the bbox
        x_max = W - bbox[:, 1]
        x_min = W - bbox[:, 3]
        bbox[:, 1] = x_min
        bbox[:, 3] = x_max
    return bbox


def random_horizflip(img):
    
    """ Randomly flip an image in vertical or horizontal direction """

    param = random.choice([True, False])
    if param:
        img = img[:,:,::-1]
    return img, param



class Transform(object):

    def __init__(self, min_size=600, max_size=1000):
        self.min_size = min_size
        self.max_size = max_size

    def __call__(self, in_data):
        image, bbox, label = in_data
        _, H, W = image.shape
        image = preprocess(image, self.min_size, self.max_size)
        _, o_H, o_W = image.shape
        scale = o_H / H
        bbox = resize_bbox(bbox, (H, W), (o_H, o_W))

        #random horizontal flip of image and bounding box
        image, param = random_horizflip(image)
        bbox = flip_bbox(bbox, (o_H, o_W), param)

        return image, bbox, label, scale


class Dataset:
    def __init__(self):
        self.db = BboxDat(voc_data_dir)
        self.tsf = Transform(min_size, max_size)

    def __getitem__(self, idx):
        ori_image, bbox, label, difficult = self.db.get_im(idx)
        image, bbox, label, scale = self.tsf((ori_image, bbox, label))
        return image.copy(), bbox.copy(), label.copy(), scale

    def __len__(self):
        return len(self.db)


class TestDataset:
    def __init__(self,split='val', use_difficult=True):
        self.db = BboxDat(voc_data_dir, split=split, use_difficult=use_difficult)

    def __getitem__(self, idx):
        ori_image, bbox, label, difficult = self.db.get_im(idx)
        image = preprocess(ori_image)
        return image, ori_image.shape[1:], bbox, label, difficult

    def __len__(self):
        return len(self.db)
