import argparse
from src import trainer
from src.utilities.utils import str2bool


if __name__ == "__main__":
    # ----------------------------------------
    #        Initialize the parameters
    # ----------------------------------------
    parser = argparse.ArgumentParser()

    # GPU parameters
    parser.add_argument('--load_name', type=str, default='',
                        help='Load to pretrained KPN model to continuously train')
    parser.add_argument('--gpu', type=str2bool, default=True,
                        help='Set True to using GPU')
    parser.add_argument('--multi_gpu', type=str2bool, default=False,
                        help='True for more than 1 GPU')
    parser.add_argument('--cudnn_benchmark', type=str2bool, default=True,
                        help='True for unchanged input data type')

    # Saving model, and loading parameters
    parser.add_argument('--save_path', type=str, default='./models',
                        help='Path to save KPN model')
    parser.add_argument('--save_name', type=str, default='KPN',
                        help='Name of KPN model')
    parser.add_argument('--sample_path', type=str, default='./samples',
                        help='Path to save derained samples')
    parser.add_argument('--separate_folder', type=str2bool, default=False,
                        help='Save sample images in separate folders or in same folder')
    parser.add_argument('--sample_encode', type=str, default='png',
                        help='Image encoder to save sample images')
    parser.add_argument('--save_by_epoch', type=int, default=10,
                        help='Interval between model checkpoints (by epochs)')
    parser.add_argument('--model_path', type=str, default='',
                        help='Load the pre-trained model with certain epoch')

    # Training parameters
    parser.add_argument('--batch_size', type=int, default=16,
                        help='Size of the batches')
    parser.add_argument('--num_workers', type=int, default=8,
                        help='Number of cpu threads to use during batch generation')
    parser.add_argument('--epochs', type=int, default=100,
                        help='Number of training epochs')
    parser.add_argument('--lr_g', type=float, default=0.0002,
                        help='Adam: learning rate for G / D')
    parser.add_argument('--b1', type=float, default=0.5,
                        help='Adam: decay of first order momentum of gradient')
    parser.add_argument('--b2', type=float, default=0.999,
                        help='Adam: decay of second order momentum of gradient')
    parser.add_argument('--weight_decay', type=float, default=0,
                        help='Weight decay for optimizer')
    parser.add_argument('--lr_decrease_epoch', type=int, default=20,
                        help='Learning rate decreases at a certain number of epochs')

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
    parser.add_argument('--init_type', type=str, default='xavier',
                        help='Initialization type of generator')
    parser.add_argument('--init_gain', type=float, default=0.02,
                        help='Initialization gain of generator')

    # Dataset parameters
    parser.add_argument('--norain_path', type=str, default='./data/test/norain',
                        help='Train norain images folder path')
    parser.add_argument('--rain_path', type=str, default='./data/test/rain',
                        help='Train rain images folder path')
    parser.add_argument('--input_size', type=int, default=224,
                        help='Batch input resize')
    parser.add_argument('--angle_aug', type=str2bool, default=False,
                        help='Geometry augmentation (rotation, flipping)')

    configs = parser.parse_args()
    print(configs)

    trainer.pretrain(configs)
