import os
import random
import numpy as np
import cv2
import math
import torch
from torch.utils.data import Dataset
from torchvision import transforms

import augmix
import utils


class RandomCrop(object):
    def __init__(self, image_size, crop_size):
        self.ch, self.cw = crop_size
        ih, iw = image_size

        self.h1 = random.randint(0, ih - self.ch)
        self.w1 = random.randint(0, iw - self.cw)

        self.h2 = self.h1 + self.ch
        self.w2 = self.w1 + self.cw

    def __call__(self, img):
        if len(img.shape) == 3:
            return img[self.h1: self.h2, self.w1: self.w2, :]
        else:
            return img[self.h1: self.h2, self.w1: self.w2]


class DenoisingDataset(Dataset):
    # root: list ; transform: torch transform
    def __init__(self, configs):
        self.configs = configs
        self.imglist = utils.get_files(configs.baseroot)

    def __getitem__(self, index):
        # read an image
        img_rainy = cv2.imread(self.imglist[index][0])
        img_gt = cv2.imread(self.imglist[index][1])

        height = img_rainy.shape[0]
        width = img_rainy.shape[1]

        height_origin = height
        width_origin = width

        if height % 16 != 0:
            height = ((height // 16) + 1) * 16

        if width % 16 != 0:
            width = ((width // 16) + 1) * 16

        img_rainy = cv2.resize(img_rainy, (width, height))
        img_gt = cv2.resize(img_gt, (width, height))

        img_rainy = cv2.cvtColor(img_rainy, cv2.COLOR_BGR2RGB)
        img_gt = cv2.cvtColor(img_gt, cv2.COLOR_BGR2RGB)

        # random crop
        if self.configs.crop:
            cropper = RandomCrop(
                img_rainy.shape[:2], (self.configs.crop_size, self.configs.crop_size))
            img_rainy = cropper(img_rainy)
            img_gt = cropper(img_gt)

        # random rotate and horizontal flip
        # two data augmentation methods are recommended
        if self.configs.angle_aug:
            rotate = random.randint(0, 3)
            if rotate != 0:
                img_rainy = np.rot90(img_rainy, rotate)
                img_gt = np.rot90(img_gt, rotate)
            if np.random.random() >= 0.5:
                img_rainy = cv2.flip(img_rainy, flipCode=0)
                img_gt = cv2.flip(img_gt, flipCode=0)

        # normalization
        img_rainy = img_rainy.astype(np.float32)  # RGB image in range [0, 255]
        img_gt = img_gt.astype(np.float32)  # RGB image in range [0, 255]
        img_rainy = img_rainy / 255.0
        img_rainy = torch.from_numpy(img_rainy.transpose(2, 0, 1)).contiguous()
        img_gt = img_gt / 255.0
        img_gt = torch.from_numpy(img_gt.transpose(2, 0, 1)).contiguous()

        return img_rainy, img_gt, height_origin, width_origin

    def __len__(self):
        return len(self.imglist)