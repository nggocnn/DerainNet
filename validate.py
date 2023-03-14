import os
import argparse
import torch
import pandas as pd
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.dataset import DeRainDataset
from src.utilities import utils
from src.utilities.utils import str2bool
from warnings import simplefilter

# Ignore all future warnings
simplefilter(action='ignore', category=FutureWarning)


if __name__ == "__main__":
    # ----------------------------------------
    #        Initialize the parameters
    # ----------------------------------------
    parser = argparse.ArgumentParser()

    # GPU parameters
    parser.add_argument('--gpu', type=str2bool, default=True,
                        help='Set True to using GPU')

    # Loading model, and parameters
    parser.add_argument('--results_path', type=str, default='./results_tmp',
                        help='Path to save the derain samples')
    parser.add_argument('--separate_folder', type=str2bool, default=True,
                        help='Save sample images in separate folders or in same folder')
    parser.add_argument('--sample_encode', type=str, default='png',
                        help='Image encoder to save sample images')
    parser.add_argument('--model_path', type=str, default='./models/model.pth',
                        help='Path of pre-trained model')
    parser.add_argument('--batch_size', type=int, default=4,
                        help='Size of the batches')
    parser.add_argument('--num_workers', type=int, default=4,
                        help='Number of cpu threads to use during batch generation')

    # Initialization parameters
    parser.add_argument('--color', type=str2bool, default=True,
                        help='Input type')
    parser.add_argument('--burst_length', type=int, default=1,
                        help='Number of photos used in burst setting')
    parser.add_argument('--blind_est', type=str2bool, default=True,
                        help='Variance map')
    parser.add_argument('--kernel_size', type=str2bool, default=[3],
                        help='Kernel size')
    parser.add_argument('--sep_conv', type=str2bool, default=False,
                        help='Simple output type')
    parser.add_argument('--channel_att', type=str2bool, default=False,
                        help='Channel wise attention')
    parser.add_argument('--spatial_att', type=str2bool, default=False,
                        help='Spatial wise attention')
    parser.add_argument('--up_mode', type=str, default='bilinear',
                        help='Up mode')
    parser.add_argument('--core_bias', type=str2bool, default=False,
                        help='Core bias')

    # Dataset parameters
    parser.add_argument('--norain_path', type=str, default='./data/test/norain',
                        help='Test norain images folder path')
    parser.add_argument('--rain_path', type=str, default='./data/test/rain',
                        help='Test rain images folder path')
    parser.add_argument('--input_size', type=int, default=224,
                        help='Batch input resize')
    parser.add_argument('--angle_aug', type=str2bool, default=False,
                        help='Geometry augmentation (rotation, flipping)')

    configs = parser.parse_args()
    print(configs)

    # Initialize
    if configs.gpu:
        generator = utils.create_generator(configs).cuda()
    else:
        generator = utils.create_generator(configs)

    test_dataset = DeRainDataset(configs)
    n_test_images = len(test_dataset)

    test_loader = DataLoader(
        test_dataset, batch_size=configs.batch_size, shuffle=False,
        num_workers=configs.num_workers, pin_memory=True
    )

    utils.check_path(configs.results_path)
    model_name = os.path.splitext(os.path.basename(configs.model_path))[0]

    psnr_sum, psnr_avg, ssim_sum, ssim_avg, idx = 0, 0, 0, 0, 0
    log_result_first_time = True
    log_result_name = model_name + '_' + utils.get_timestamp() + '.csv'
    log_result_path = os.path.join(configs.results_path, log_result_name)

    # Forward
    for i, (rain_input, norain_input, origin_height, origin_width, names) \
            in enumerate(tqdm(test_loader, desc=f'Evaluating...')):

        # To device
        if configs.gpu and torch.cuda.is_available():
            rain_input = rain_input.cuda()
            norain_input = norain_input.cuda()

        # Forward propagation
        with torch.no_grad():
            derain_output = generator(rain_input, rain_input)

        derain_output = rain_input

        for j in range(len(rain_input)):
            # Save
            titles = ['rain', 'norain', 'derain']
            images = [rain_input[j], norain_input[j], derain_output[j]]
            utils.save_sample(
                folder=configs.results_path, titles=titles, name=names[j],
                images=images, height=origin_height[j], width=origin_width[j],
                separate_folder=configs.separate_folder, encode=configs.sample_encode
            )

            # Evaluation
            derain_image = utils.recover_image(derain_output[j], height=origin_height[j], width=origin_width[j])
            norain_image = utils.recover_image(norain_input[j], height=origin_height[j], width=origin_width[j])

            psnr_ = psnr(derain_image, norain_image, data_range=255)
            psnr_sum += psnr_

            ssim_ = ssim(derain_image, norain_image, data_range=255, multichannel=True)
            ssim_sum += ssim_

            # Logging
            evaluation_results = pd.DataFrame({
                'image': names[j],
                'psnr': psnr_,
                'ssim': ssim_
            }, index=[idx])

            if log_result_first_time:
                evaluation_results.to_csv(log_result_path, mode='w', header=True, index=True, float_format='%.6f')
                log_result_first_time = False
            else:
                evaluation_results.to_csv(log_result_path, mode='a', header=False, index=True, float_format='%.6f')

            idx += 1

    psnr_avg = psnr_sum / n_test_images
    ssim_avg = ssim_sum / n_test_images

    evaluation_results = pd.DataFrame({
        'image': 'average',
        'psnr': psnr_avg,
        'ssim': ssim_avg
    }, index=[idx])

    evaluation_results.to_csv(log_result_path, mode='a', header=False, index=True, float_format='%.6f')
