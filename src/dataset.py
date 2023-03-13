import random

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

from src.utilities import utils


class DeNoiseDataset(Dataset):
    def __init__(self, configs):
        self.configs = configs
        self.image_list, self.name_list = utils.get_files(configs.base_root)
        self.input_size = configs.input_size

    def __getitem__(self, index):
        # read an image
        rainy_images = cv2.imread(self.image_list[index][0])
        norain_images = cv2.imread(self.image_list[index][1])

        height_origin = rainy_images.shape[0]
        width_origin = rainy_images.shape[1]

        height = width = self.input_size

        if self.input_size % 16 != 0:
            height = width = ((self.input_size // 16) + 1) * 16

        rainy_images = cv2.resize(rainy_images, (width, height))
        norain_images = cv2.resize(norain_images, (width, height))

        # rainy_images = cv2.cvtColor(rainy_images, cv2.COLOR_BGR2RGB)
        # norain_images = cv2.cvtColor(norain_images, cv2.COLOR_BGR2RGB)

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

        # normalization
        rainy_images = rainy_images.astype(np.float32)  # pixel value in range [0, 255]
        rainy_images = rainy_images / 255.0
        rainy_images = torch.from_numpy(rainy_images.transpose(2, 0, 1)).contiguous()
        norain_images = norain_images.astype(np.float32)  # pixel value image in range [0, 255]
        norain_images = norain_images / 255.0
        norain_images = torch.from_numpy(norain_images.transpose(2, 0, 1)).contiguous()

        return rainy_images, norain_images, height_origin, width_origin, self.name_list[index]

    def __len__(self):
        return len(self.image_list)
