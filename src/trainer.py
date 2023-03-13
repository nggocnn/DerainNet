import os

import torch
from torch import nn
from torch.backends import cudnn
from torch.utils.data import DataLoader
from tqdm import tqdm

from src import dataset
from src import ssim
from src.utilities import utils


def pretrain(configs):
	cudnn.benchmark = configs.cudnn_benchmark

	# configurations
	save_folder = configs.save_path
	sample_folder = configs.sample_path
	utils.check_path(save_folder)
	utils.check_path(sample_folder)

	# Loss functions
	if configs.gpu and torch.cuda.is_available():
		criterion_L1 = torch.nn.L1Loss().cuda()
		criterion_L2 = torch.nn.MSELoss().cuda()
		criterion_ssim = ssim.SSIM().cuda()

	# Initialize Generator
	generator = utils.create_generator(configs)

	# To device
	if configs.gpu and torch.cuda.is_available():
		if configs.multi_gpu:
			generator = nn.DataParallel(generator)
			generator = generator.cuda()
		else:
			generator = generator.cuda()

	# Optimizers
	optimizer = torch.optim.Adam(
		filter(lambda p: p.requires_grad, generator.parameters()),
		lr=configs.lr_g,
		betas=(configs.b1, configs.b2),
		weight_decay=configs.weight_decay
	)

	gpu_num = torch.cuda.device_count()
	print(f'Using {gpu_num} GPU(s)')

	# Define the dataset
	trainset = dataset.DeNoiseDataset(configs)

	# Define the dataloader
	train_loader = DataLoader(
		trainset, batch_size=configs.batch_size, shuffle=True,
		num_workers=configs.num_workers, pin_memory=True)

	# ----------------------------------------
	#                 Training
	# ----------------------------------------

	# For loop training
	for epoch in range(configs.epochs):
		for i, (rainy_input, norain_input, origin_height, origin_width, name) \
				in enumerate(tqdm(train_loader, desc=f'Epoch: {epoch + 1}/{configs.epochs}')):

			# To device
			if configs.gpu and torch.cuda.is_available():
				rainy_input = rainy_input.cuda()
				norain_input = norain_input.cuda()

			# Train Generator
			optimizer.zero_grad()
			derain_output = generator(rainy_input, rainy_input)
			ssim_loss = -criterion_ssim(norain_input, derain_output)

			PixelLevel_L1_Loss = criterion_L1(derain_output, norain_input)
			# PixelLevel_L2_Loss = criterion_L2(fake_target, true_target)

			# Overall Loss and optimize
			loss = PixelLevel_L1_Loss + 0.2 * ssim_loss

			loss.backward()
			optimizer.step()

		# Save model at certain epochs or iterations
		save_model(configs, (epoch + 1), generator)

		# Learning rate decrease at certain epochs
		adjust_learning_rate(configs, (epoch + 1), optimizer)

		# Sample data every epoch
		# if (epoch + 1) % 1 == 0:
		# 	img_list = [rainy_input, derain_output, norain_input]
		# 	name_list = ['in', 'pred', 'gt']
		# 	# utilities.save_sample(folder=sample_folder, sample_name='train_epoch%d' % (epoch + 1), img_list=img_list,
		# 	#                   name_list=name_list, pixel_max_cnt=255)


def adjust_learning_rate(configs, epoch, optimizer):
	target_epoch = configs.epochs - configs.lr_decrease_epoch
	remain_epoch = configs.epochs - epoch

	if epoch >= configs.lr_decrease_epoch:
		lr = configs.lr_g * remain_epoch / target_epoch
		for param_group in optimizer.param_groups:
			param_group['lr'] = lr


def save_model(configs, epoch, generator):
	"""
	Save the model at "checkpoint_interval" and its multiple
	"""

	# Define the name of trained model
	model_name = f'KPN_rainy_image_epoch_{epoch}.pth'
	save_model_path = os.path.join(configs.save_path, model_name)

	if (epoch % configs.save_by_epoch == 0):
		if configs.multi_gpu:
			torch.save(generator.module.state_dict(), save_model_path)
		else:
			torch.save(generator.state_dict(), save_model_path)

		print(f'Model is successfully saved at epoch {epoch}' % epoch)
