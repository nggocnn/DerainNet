import os
import pandas as pd
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
	utils.check_path(configs.save_path)
	utils.check_path(configs.sample_path)

	# Loss functions
	if configs.gpu and torch.cuda.is_available():
		criterion_L1 = torch.nn.L1Loss().cuda()
		# criterion_L2 = torch.nn.MSELoss().cuda()
		criterion_ssim = ssim.SSIM().cuda()
	else:
		criterion_L1 = torch.nn.L1Loss()
		# criterion_L2 = torch.nn.MSELoss()
		criterion_ssim = ssim.SSIM()

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
	train_dataset = dataset.DeRainDataset(configs)
	n_train_images = len(train_dataset)

	# Define the dataloader
	train_loader = DataLoader(
		train_dataset, batch_size=configs.batch_size, shuffle=True,
		num_workers=configs.num_workers, pin_memory=True)

	# ----------------------------------------
	#                 Training
	# ----------------------------------------

	log_result_first_time = True
	log_result_name = configs.save_name + '_loss_log_' + utils.get_timestamp() + '.csv'
	log_result_path = os.path.join(configs.save_path, log_result_name)

	# For loop training
	rain_input, norain_input, derain_output, origin_height, origin_width, names = None, None, None, None, None, None

	for epoch in range(configs.epochs):
		l1_loss_sum, l1_loss_avg, ssim_loss_sum, ssim_loss_avg, loss_sum, loss_avg = 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
		for i, (rain_input, norain_input, origin_height, origin_width, names) \
			in enumerate(tqdm(train_loader, desc=f'Epoch: {epoch + 1}/{configs.epochs}')):

			# To device
			if configs.gpu and torch.cuda.is_available():
				rain_input = rain_input.cuda()
				norain_input = norain_input.cuda()

			# Train Generator
			optimizer.zero_grad()
			derain_output = generator(rain_input, rain_input)

			# Overall Loss and optimize
			ssim_loss = criterion_ssim(norain_input, derain_output)
			l1_loss = criterion_L1(norain_input, derain_output)
			loss = l1_loss - 0.2 * ssim_loss
			loss.backward()
			optimizer.step()

			l1_loss_sum += float(l1_loss.clone().data.cpu()) * len(rain_input)
			ssim_loss_sum += float(ssim_loss.clone().data.cpu()) * len(rain_input)
			loss_sum += float(loss.clone().data.cpu()) * len(rain_input)

		l1_loss_avg = l1_loss_sum / n_train_images
		ssim_loss_avg = ssim_loss_sum / n_train_images
		loss_avg = loss_sum / n_train_images

		evaluation_results = pd.DataFrame({
			'epoch': epoch,
			'l1_loss': l1_loss_avg,
			'ssim_loss': ssim_loss_avg,
			'total_loss': loss_avg
		}, index=[epoch])

		if log_result_first_time:
			evaluation_results.to_csv(log_result_path, mode='w', header=True, index=True, float_format='%.6f')
			log_result_first_time = False
		else:
			evaluation_results.to_csv(log_result_path, mode='a', header=False, index=True, float_format='%.6f')

		# Save model at certain epochs or iterations
		save_model(configs, (epoch + 1), generator)

		# Learning rate decrease at certain epochs
		adjust_learning_rate(configs, (epoch + 1), optimizer)

		# Save sample data
		titles = ['rain', 'norain', 'derain']
		images = [rain_input[0], norain_input[0], derain_output[0]]
		utils.save_sample(
			folder=configs.sample_path, titles=titles, name=names[0],
			images=images, height=origin_height[0], width=origin_width[0],
			separate_folder=configs.separate_folder, encode=configs.sample_encode
		)


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
	if epoch % configs.save_by_epoch == 0:
		# Define the name of trained model
		model_name = f'{configs.save_name}_epoch_{epoch}.pth'
		save_model_path = os.path.join(configs.save_path, model_name)

		if configs.multi_gpu:
			torch.save(generator.module.state_dict(), save_model_path)
		else:
			torch.save(generator.state_dict(), save_model_path)

		print(f'Model is successfully saved at epoch {epoch}: {save_model_path}')
