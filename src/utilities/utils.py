import math
import os

import cv2
import numpy as np
import skimage
import torch
import argparse
from datetime import datetime

from src import efderainnet
from src.utilities.constants import IMAGE_EXTENSION


def create_generator(configs):
    generator = efderainnet.KPN(
        configs.color, configs.burst_length, configs.blind_est,
        configs.kernel_size, configs.sep_conv, configs.channel_att,
        configs.spatial_att, configs.up_mode, configs.core_bias
    )

    if configs.model_path == '':
        efderainnet.weights_init(generator, init_type=configs.init_type, init_gain=configs.init_gain)
        print('Generator is created!')
    else:
        pretrained_net = torch.load(configs.model_path)
        load_dict(generator, pretrained_net)
        print(f'Generator is loaded! {configs.model_path}')

    return generator


def load_dict(train_net, pretrained_net):
    # Get the dict from pre-trained network
    pretrained_dict = pretrained_net
    # Get the dict from processing network
    process_dict = train_net.state_dict()
    # Delete the extra keys of pretrained_dict that do not belong to process_dict
    pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in process_dict}
    # Update train_net using pretrained_dict
    process_dict.update(pretrained_dict)
    # Load the updated dict to processing network
    train_net.load_state_dict(process_dict)
    return train_net


def save_sample(folder, titles, name, images, height=0, width=0, separate_folder=True, encode='png'):
    assert len(titles) == len(images), \
        f'Titles list (len: {len(titles)}) and Images list (len: {len(images)}) must have same length'

    if encode not in IMAGE_EXTENSION:
        raise Exception(f'Image extension {encode} is not supported')

    for i in range(len(images)):
        # Save image to path
        save_path = ''
        if separate_folder:
            save_path = os.path.join(folder, titles[i])
            check_path(save_path)
            save_path = os.path.join(save_path, name + '.' + encode)
        else:
            save_path = os.path.join(folder, name + '_' + titles[i] + '.' + encode)
        cv2.imwrite(save_path, recover_image(images[i], height, width))


def recover_image(image, height=0, width=0):
    # Recover normalization
    image = image * 255.0

    # Process image_copy and do not destroy the data of image
    image_copy = image.clone().data.permute(1, 2, 0).cpu().numpy()
    image_copy = np.clip(image_copy, 0, 255.0).astype(np.uint8)

    if height > 0 and height > 0:
        image_copy = cv2.resize(image_copy, (int(width), int(height)))

    return image_copy


def psnr(pred, target):
    mse = np.mean((pred - target) ** 2)
    if mse == 0:
        return 100
    PIXEL_MAX = 255.0
    return 20 * math.log10(PIXEL_MAX / math.sqrt(mse))


def grey_psnr(pred, target, pixel_max_cnt=255):
    pred = torch.sum(pred, dim=0)
    target = torch.sum(target, dim=0)
    mse = torch.mul(target - pred, target - pred)
    rmse_avg = (torch.mean(mse).item()) ** 0.5
    p = 20 * np.log10(pixel_max_cnt * 3 / rmse_avg)
    return p


def ssim(img1, img2):
    img1 = img1.clone().data.permute(0, 2, 3, 1).cpu().numpy()
    img2 = img2.clone().data.permute(0, 2, 3, 1).cpu().numpy()
    img2 = img2[0]
    img1 = img1[0]
    return skimage.measure.compare_ssim(img2, img1, multichannel=True)


def check_path(path):
    if not os.path.exists(path):
        os.makedirs(path)


def get_timestamp(timestamp_format="%Y%m%d-%H%M%S"):
    return datetime.now().strftime(timestamp_format)


def check_file_extension(file_path: str, extension_type: str):
    return file_path.endswith(extension_type)


def text_save(content, filename, mode='a'):
    # Save a list to a txt
    # Try to save a list variable in txt file.
    file = open(filename, mode)
    for i in range(len(content)):
        file.write(str(content[i]))
    file.close()


def get_files(rain_path, norain_path, image_encoder='jpg'):

    if not os.path.exists(rain_path):
        raise Exception(f'Data path {rain_path} is not valid!')
    if not os.path.exists(norain_path):
        raise Exception(f'Data path {norain_path} is not valid!')

    results = []
    names = []

    for file in os.listdir(rain_path):
        if file.split('.')[1] != image_encoder:
            continue

        rainy_image = rain_path + "/" + file
        norain_image = norain_path + "/" + file

        if not os.path.exists(rainy_image):
            print(f'Image file {rainy_image} is not valid!')
        if not os.path.exists(norain_image):
            print(f'Data path {norain_image} is not valid!')

        results.append([rainy_image, norain_image])
        names.append(file.split('.')[0])

    if len(results) == 0:
        raise Exception('No image has been loaded!')
    else:
        print(f'{len(results)} pairs of rain and norain images have been loaded!')

    return results, names


def str2bool(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Unsupported value encountered.')
