import random

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

from src.utilities import utils


class DeRainDataset(Dataset):
    def __init__(self, configs):
        self.configs = configs
        self.image_list, self.name_list = \
            utils.get_files(rain_path=configs.rain_path, norain_path=configs.norain_path)
        self.input_size = configs.input_size

    def __getitem__(self, index):
        # read an image
        rainy_images = cv2.imread(self.image_list[index][0])
        norain_images = cv2.imread(self.image_list[index][1])

        origin_height = rainy_images.shape[0]
        origin_width = rainy_images.shape[1]

        if self.input_size > 0:
            height = width = (self.input_size // 16) * 16
            rainy_images = cv2.resize(rainy_images, (width, height))
            norain_images = cv2.resize(norain_images, (width, height))

        # Two data augmentation methods are recommended
        # Random rotate and Horizontal flip
        if self.configs.angle_aug:
            rotate = random.randint(0, 3)
            if rotate != 0:
                rainy_images = np.rot90(rainy_images, rotate)
                norain_images = np.rot90(norain_images, rotate)
            if np.random.random() >= 0.5:
                rainy_images = cv2.flip(rainy_images, flipCode=0)
                norain_images = cv2.flip(norain_images, flipCode=0)

        # Normalization
        rainy_images = rainy_images.astype(np.float32)
        rainy_images = rainy_images / 255.0
        rainy_images = torch.from_numpy(rainy_images.transpose(2, 0, 1)).contiguous()
        norain_images = norain_images.astype(np.float32)
        norain_images = norain_images / 255.0
        norain_images = torch.from_numpy(norain_images.transpose(2, 0, 1)).contiguous()

        return rainy_images, norain_images, origin_height, origin_width, self.name_list[index]

    def __len__(self):
        return len(self.image_list)
