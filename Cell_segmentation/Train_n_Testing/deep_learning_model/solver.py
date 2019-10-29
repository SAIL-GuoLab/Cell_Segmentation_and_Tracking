import os
from glob import glob
import numpy as np
import time
import datetime
import torch
import torchvision
from torch import optim
from torch.autograd import Variable
from torch.nn import BCELoss, SmoothL1Loss
import torch.nn.functional as F
from network import U_Net, R2U_Net, AttU_Net, R2AttU_Net, ResAttU_Net
from evaluation import CrossEntropy
import csv
import timeit
from scipy import ndimage
import scipy.ndimage.morphology as ndi_morph
import skimage.morphology as skimage_morph
from skimage import io
import pandas as pd

# Additional imports (R&R)
from matplotlib import pyplot as plt

class Solver(object):
	def __init__(self, config, train_loader, validation_loader, test_loader):

		# Data loader
		self.train_loader = train_loader
		self.validation_loader = validation_loader
		self.test_loader = test_loader
		self.down_factor = config.down_factor

		# Models
		self.unet = None
		self.optimizer = None
		self.img_ch = config.img_ch
		self.GT_ch = config.GT_ch
		self.output_ch = config.output_ch
		self.first_layer_numKernel = config.first_layer_numKernel
		self.UnetLayer = config.UnetLayer

		# Hyper-parameters
		self.initial_lr = config.lr
		self.current_lr = config.lr
		self.patch_num = config.patch_num // config.down_factor ** 2
        
		self.optimizer_choice = config.optimizer_choice
		if config.optimizer_choice == 'Adam':
			self.beta1 = config.beta1
			self.beta2 = config.beta2
		elif config.optimizer_choice == 'SGD':
			self.momentum = config.momentum
		else:
			print('No such optimizer available')

		# Loss Function
		if config.loss_function == 'BCE':
			self.loss_function_name = 'BCE'
			self.loss_function = BCELoss()
		elif config.loss_function == 'SmoothL1':
			self.loss_function_name = 'SmoothL1'
			self.loss_function = SmoothL1Loss()
		elif config.loss_function == 'Dice':
			self.loss_function_name = 'Dice'

		# Training settings
		self.num_epochs = config.num_epochs
		self.batch_size = config.batch_size
		self.withTF = config.withTF
		self.edge_enhance = config.edge_enhance

		# Early stop or not
		self.early_stop = config.early_stop

		# Path
		self.current_model_saving_path = config.current_model_saving_path
		self.current_prediction_path = config.current_prediction_path
		self.current_loss_history_path = config.current_loss_history_path
		self.test_result_comparison_path = config.test_result_comparison_path
		self.test_GT_list = config.GT_test
		self.mode = config.mode

		self.device = torch.device('cuda: %d' % config.cuda_idx)
		self.model_type = config.model_type
		self.t = config.t
		self.build_model()

	def build_model(self):
		"""Build generator and discriminator."""
		if self.model_type =='U_Net':
			self.unet = U_Net(UnetLayer = self.UnetLayer, img_ch = self.img_ch, output_ch = self.output_ch, first_layer_numKernel = self.first_layer_numKernel)
		elif self.model_type =='R2U_Net':
			self.unet = R2U_Net(img_ch = self.img_ch, output_ch = self.output_ch, t = self.t, first_layer_numKernel = self.first_layer_numKernel)
		elif self.model_type =='AttU_Net':
			self.unet = AttU_Net(img_ch = self.img_ch, output_ch = self.output_ch, first_layer_numKernel = self.first_layer_numKernel)
		elif self.model_type == 'R2AttU_Net':
			self.unet = R2AttU_Net(img_ch = self.img_ch, output_ch = self.output_ch, t = self.t, first_layer_numKernel = self.first_layer_numKernel)
		elif self.model_type == 'ResAttU_Net':
			self.unet = ResAttU_Net(UnetLayer = self.UnetLayer, img_ch = self.img_ch, output_ch = self.output_ch, first_layer_numKernel = self.first_layer_numKernel)
			
		if self.optimizer_choice == 'Adam':
			self.optimizer = optim.Adam(list(self.unet.parameters()), self.initial_lr, [self.beta1, self.beta2])
		elif self.optimizer_choice == 'SGD':
			self.optimizer = optim.SGD(list(self.unet.parameters()), self.initial_lr, self.momentum)
		else:
			pass

		self.unet.to(self.device)

	def print_network(self, model, name):
		"""Print out the network information."""
		num_params = 0
		for p in model.parameters():
			num_params += p.numel()
		print(model)
		print(name)
		print("The number of parameters: {}".format(num_params))
		
	def to_data(self, x):
		"""Convert variable to tensor."""
		if torch.cuda.is_available():
			x = x.cpu()
		return x.data

	# Redefine the 'update_lr' function (R&R)
	def update_lr(self, new_lr):
		for param_group in self.optimizer.param_groups:
			param_group['lr'] = new_lr          

	# Define adaptive learning rate handler (R&R)
	# This only works for non-negative loss.
	def adaptive_lr_handler(self, cooldown, min_lr, current_epoch, previous_update_epoch, plateau_ratio, adjustment_ratio, loss_history):
		if current_epoch > 1:
			if current_epoch - previous_update_epoch > cooldown:
				if (loss_history[-1] > loss_history[-2]) or (loss_history[-1]/loss_history[-2] > plateau_ratio):
					if self.current_lr > min_lr:
						self.current_lr = adjustment_ratio * self.current_lr
						self.update_lr(self.current_lr)
						print('Validation loss stop decreasing. Adjust the learning rate to {}.'.format(self.current_lr))
						return current_epoch

	def reset_grad(self):
		"""Zero the gradient buffers."""
		self.unet.zero_grad()

	def dice_coeff_loss(self, Prediction_vector, GT_vector):
		smooth = 1
		intersection = (GT_vector * Prediction_vector).sum()
		return 1 - (2. * intersection + smooth) / (GT_vector.sum() + Prediction_vector.sum() + smooth)

	def train(self):
		"""Train encoder, generator and discriminator."""

		#====================================== Training ===========================================#
		#===========================================================================================#

		unet_path = os.path.join(self.current_model_saving_path, '%s-%s-%.4f-%d-%d-%d-best.pkl' %(self.model_type, self.optimizer_choice, self.initial_lr, self.num_epochs, self.batch_size, self.down_factor))
		last_unet_path = os.path.join(self.current_model_saving_path, '%s-%s-%.4f-%d-%d-%d-last.pkl' %(self.model_type, self.optimizer_choice, self.initial_lr, self.num_epochs, self.batch_size, self.down_factor))
		print('The U-Net path is {}'.format(unet_path))
		# U-Net Train
		# Train loss history (R&R)
		train_loss_history = []
		# Validation loss history (R&R)
		validation_loss_history = []

		if os.path.isfile(unet_path):
			# Load the pretrained Encoder
			self.unet.load_state_dict(torch.load(unet_path))
			print('%s is Successfully Loaded from %s'  %(self.model_type,unet_path))

		else:
			# Train for Encoder
			best_unet_score = 0.
			print('Start training. The initial learning rate is: {}'.format(self.initial_lr))

			# Write the first line of the train and validation loss history csv file.
			with open(os.path.join(self.current_loss_history_path, 'train_and_validation_history.csv'), 'a', \
					encoding = 'utf-8', newline= '') as f:
				wr = csv.writer(f)
				wr.writerow(['Mode', 'Current Epoch', 'Total Epoch', 'Batch Size', 'Metric', 'Loss'])
				f.close()

			for epoch in range(self.num_epochs):
				self.unet.train(True)
				train_epoch_loss = 0; validation_epoch_loss = 0

				length = 0
				start_time = timeit.default_timer()

				for batch, (img, GT) in enumerate(self.train_loader):
					img = img.to(self.device)
					GT = GT.to(self.device)

					# Reshape the images and GTs to 4-dimensional so that they can get fed to the conv2d layer. (R&R)
					# The new shape has to be (batch_size, num_channels, img_dim1, img_dim2).
					if self.img_ch == 1:
						img = img[:, np.newaxis, :, :]
					else:
						img = img.transpose(1, 3); img = img.transpose(2, 3)

					if self.GT_ch == 1:
						GT = GT[:, np.newaxis, :, :]
					else:
						GT = GT.transpose(1, 3); GT = GT.transpose(2, 3)

					#  SR : Segmentation Result
					SR = torch.sigmoid(self.unet(img))

					# Flatten the prediction, target, and training field.
					SR_flat = SR.view(SR.size(0), -1)
					GT_flat = GT.view(GT.size(0), -1)

					# Compute the loss for this batch.
					if self.loss_function_name == 'Dice':
						train_loss = self.dice_coeff_loss(SR_flat, GT_flat)
					else:
						train_loss = self.loss_function(SR_flat, GT_flat)

					# Add the loss of this batch to the loss of this epoch.              
					train_epoch_loss += train_loss.item()
					if self.edge_enhance == 'True':
						GT_edge_enhanced = ndimage.gaussian_laplace(np.squeeze(GT.cpu().detach().numpy()), sigma = 5)
						GT_edge1 = torch.tensor(np.int64(GT_edge_enhanced < -0.001))
						GT_edge2 = torch.tensor(np.int64(GT_edge_enhanced > 0.001))
						y_hat = torch.cat((torch.squeeze(SR_flat), torch.squeeze(SR)[GT_edge1 == 1], torch.squeeze(SR)[GT_edge2 == 1]), 0)
						y = torch.cat((torch.squeeze(GT_flat), torch.squeeze(GT)[GT_edge1 == 1], torch.squeeze(GT)[GT_edge2 == 1]), 0)
					elif self.edge_enhance == 'Double':
						GT_edge_enhanced = ndimage.gaussian_laplace(np.squeeze(GT.cpu().detach().numpy()), sigma = 5)
						GT_edge1 = torch.tensor(np.int64(GT_edge_enhanced < -0.001))
						GT_edge2 = torch.tensor(np.int64(GT_edge_enhanced > 0.001))
						y_hat = torch.cat((torch.squeeze(SR_flat), torch.squeeze(SR)[GT_edge1 == 1], torch.squeeze(SR)[GT_edge1 == 1], torch.squeeze(SR)[GT_edge2 == 1]), 0)
						y = torch.cat((torch.squeeze(GT_flat), torch.squeeze(GT)[GT_edge1 == 1], torch.squeeze(GT)[GT_edge1 == 1], torch.squeeze(GT)[GT_edge2 == 1]), 0)
		
					train_loss = self.loss_function(y_hat, y)

					# Backprop + optimize
					self.reset_grad()
					train_loss.backward()
					self.optimizer.step()
                    
                    ### if batch size = 1  ###
					length += 1

					if batch % 200 == 0:
						print('[Training] Epoch [{}/{}], Batch: {}, Batch size: {}, Average {} Error: {}'.format(epoch + 1, self.num_epochs, batch, self.batch_size, self.loss_function_name, train_epoch_loss/length))


 					# Empty cache to free up memory at the end of each batch.
					del batch, img, GT, SR, GT_flat, SR_flat, train_loss
					torch.cuda.empty_cache() 

				end_time = timeit.default_timer()
				# Normalize the train loss over the length of the epoch (number of images in this epoch).
				train_epoch_loss = train_epoch_loss/length

				# Print the log info
				print('[Training] Epoch [%d/%d], Train Loss: %.6f, Run Time: %.4f [h]' % (epoch + 1, self.num_epochs, train_epoch_loss, (end_time - start_time) / 60 / 60))

				# Append train loss to train loss history (R&R)
				train_loss_history.append(train_epoch_loss)
				with open(os.path.join(self.current_loss_history_path, 'train_and_validation_history.csv'), 'a', \
                         encoding = 'utf-8', newline= '') as f:
					wr = csv.writer(f)
					wr.writerow(['Training', '%d' % (epoch + 1), '%d' % (self.num_epochs), '%d' % (self.batch_size), \
                         	     '%s' % self.loss_function_name, '%.6f' % train_epoch_loss])
					f.close()

				#===================================== Validation ====================================#
				self.unet.train(False)
				self.unet.eval()

				length = 0
				start_time = timeit.default_timer()

				for batch, (img, GT) in enumerate(self.validation_loader):
					# Read, reshape the GTs and images, and compute the target images.
					img = img.to(self.device)
					GT = GT.to(self.device)

					# Reshape the images and GTs to 4-dimensional so that they can get fed to the conv2d layer. (R&R)
					# The new shape has to be (batch_size, num_channels, img_dim1, img_dim2).
					if self.img_ch == 1:
						img = img[:, np.newaxis, :, :]
					else:
						img = img.transpose(1, 3); img = img.transpose(2, 3)

					if self.GT_ch == 1:
						GT = GT[:, np.newaxis, :, :]
					else:
						GT = GT.transpose(1, 3); GT = GT.transpose(2, 3)

					#  SR : Segmentation Result
					SR = torch.sigmoid(self.unet(img))
        
					# Flatten the prediction and target.
					SR_flat = SR.view(SR.size(0), -1)
					GT_flat = GT.view(GT.size(0), -1)
                        
					# Compute the loss for this batch.
					if self.loss_function_name == 'Dice':
						validation_loss = self.dice_coeff_loss(SR_flat, GT_flat)
					else:
						validation_loss = self.loss_function(SR_flat, GT_flat)
					length += 1
					validation_epoch_loss += validation_loss.item()
                    
 
					# Empty cache to free up memory at the end of each batch.
					del img, GT, SR, GT_flat, SR_flat, validation_loss
					torch.cuda.empty_cache() 

				# Normalize the validation loss.
				validation_epoch_loss = validation_epoch_loss/length
                
				end_time = timeit.default_timer()

				# Define the decisive score of the network as 1 - validation_epoch_loss.
				unet_score = 1. - validation_epoch_loss
				print('Current learning rate: {}'.format(self.current_lr))

				print('[Validation] Epoch [%d/%d] Validation Loss: %.6f, Run Time: %.4f [h]' % (epoch + 1, self.num_epochs, validation_epoch_loss, (end_time - start_time)/60/60))

				# Append validation loss to train loss history (R&R)
				validation_loss_history.append(validation_epoch_loss)				
				with open(os.path.join(self.current_loss_history_path, 'train_and_validation_history.csv'), 'a', \
                         encoding = 'utf-8', newline= '') as f:
					wr = csv.writer(f)
					wr.writerow(['Validation', '%d' % (epoch + 1), '%d' % (self.num_epochs), '%d' % (self.batch_size), \
                             	 '%s' % self.loss_function_name, '%.6f' % validation_epoch_loss])
					f.close()

				# Make sure we save the best and last unets.
				if unet_score > best_unet_score:
					best_unet_score = unet_score
					best_epoch = epoch
					best_unet = self.unet.state_dict()
					print('Best %s model score : %.6f' % (self.model_type, best_unet_score))
					torch.save(best_unet, unet_path)
				if (epoch == self.num_epochs - 1):
					last_unet = self.unet.state_dict()
					torch.save(last_unet, last_unet_path)
				if epoch % 10 == 0 and epoch != 0:
					epoch_unet_path_component = unet_path.split('/')
					file_name = epoch_unet_path_component[-1].replace('best', 'epoch%d' % epoch)
					epoch_unet_path_component[-1] = file_name
					epoch_unet_path = '/'.join(epoch_unet_path_component)
					epoch_unet = self.unet.state_dict()
					torch.save(epoch_unet, epoch_unet_path)

				# Adaptive Learning Rate (R&R)
				try:
					previous_epoch = self.adaptive_lr_handler(3, 0.01*self.initial_lr, epoch, previous_epoch, 0.98, 0.5, validation_loss_history)
				except:
					previous_epoch = self.adaptive_lr_handler(3, 0.01*self.initial_lr, epoch, 0, 0.98, 0.5, validation_loss_history)
				
				# Early stop (R&R)
				if (self.early_stop == True):
					if (len(validation_loss_history) > 9):
						if (np.mean(validation_loss_history[-10:-5]) <= np.mean(validation_loss_history[-5:])):
							print('Validation loss stop decreasing. Stop training.')
							last_unet = self.unet.state_dict()
							torch.save(last_unet, last_unet_path)
							break        
       
		del self.unet
		try:
			del best_unet
			torch.cuda.empty_cache()
		except:
			print('Cannot delete the variable "best_unet": variable does not exist.')
        
		return train_loss_history, validation_loss_history


	def test(self, which_unet = 'best', stop_epoch = None):

		"""Test encoder, generator and discriminator."""
		#======================================= Test ====================================#
		#=================================================================================#
		unet_path = os.path.join(self.current_model_saving_path, '%s-%s-%.4f-%d-%d-%d-best.pkl' %(self.model_type, self.optimizer_choice, self.initial_lr, self.num_epochs, self.batch_size, self.down_factor))
		last_unet_path = os.path.join(self.current_model_saving_path, '%s-%s-%.4f-%d-%d-%d-last.pkl' %(self.model_type, self.optimizer_choice, self.initial_lr, self.num_epochs, self.batch_size, self.down_factor))
		

		self.build_model()
		if not os.path.exists(self.current_prediction_path):
			os.makedirs(self.current_prediction_path)
		if stop_epoch == None:
			if which_unet == 'best':
				self.unet.load_state_dict(torch.load(unet_path))
				save_folder = 'best/'
			elif which_unet == 'last':
				self.unet.load_state_dict(torch.load(last_unet_path))
				save_folder = 'last/'
			else:
				print('Input argument which_unet must be either "best" or "last"')
		else:
			epoch_unet_path_component = unet_path.split('/')
			file_name = epoch_unet_path_component[-1].replace('best', 'epoch%d' % stop_epoch)
			epoch_unet_path_component[-1] = file_name
			epoch_unet_path = '/'.join(epoch_unet_path_component)
			self.unet.load_state_dict(torch.load(epoch_unet_path))
			save_folder = 'epoch%d/' % stop_epoch
		if not os.path.exists(self.current_prediction_path + save_folder):
			os.makedirs(self.current_prediction_path + save_folder)

		self.unet.train(False)
		self.unet.eval()
		length = 0
		test_epoch_loss = 0
		test_GT_list = self.test_GT_list

		for batch, (img, GT) in enumerate(self.test_loader):
			img = img.to(self.device)
			GT = GT.to(self.device)

			# Reshape the images and GTs to 4-dimensional so that they can get fed to the conv2d layer. (R&R)
			# The new shape has to be (batch_size, num_channels, img_dim1, img_dim2).
			if self.img_ch == 1:
				img = img[:, np.newaxis, :, :]
			else:
				img = img.transpose(1, 3); img = img.transpose(2, 3)

			if self.GT_ch == 1:
				GT = GT[:, np.newaxis, :, :]
			else:
				GT = GT.transpose(1, 3); GT = GT.transpose(2, 3)

			#plt.subplot(1,2,1); plt.imshow(img[0, 0, :, :].cpu().detach().numpy())
			#plt.subplot(1,2,2); plt.imshow(GT[0, 0, :, :].cpu().detach().numpy())

			#  SR : Segmentation Result
			SR = torch.sigmoid(self.unet(img))
        
			# Flatten the prediction and target.
			SR_flat = SR.view(SR.size(0), -1)
			GT_flat = GT.view(GT.size(0), -1)

			# Compute test loss
			if self.loss_function_name == 'Dice':
				test_loss = self.dice_coeff_loss(SR_flat, GT_flat)
			else:
				test_loss = self.loss_function(SR_flat, GT_flat)
			np_img = np.squeeze(SR.cpu().detach().numpy()) 

			# extract filename from test folder
			filename = test_GT_list[batch//self.patch_num][-15:-4]

			length += 1
			test_epoch_loss += test_loss.item()

			np.save(self.current_prediction_path + save_folder + filename + '_' + str(batch % self.patch_num).zfill(2) + '_modelprediction' + '.npy', np_img)

			del batch, img, GT, SR, GT_flat, SR_flat, test_loss
			torch.cuda.empty_cache()

		test_epoch_loss = test_epoch_loss/length
		print('Model type: ', self.model_type, 'Test loss: ', test_epoch_loss)

		# Store the test result together with all other ones for comparison
		# If the file doesn't exist or is empty, write the header.
		if os.path.exists(os.path.join(self.test_result_comparison_path, 'test_result_comparison.csv')):
			df = pd.read_csv(os.path.join(self.test_result_comparison_path, 'test_result_comparison.csv'))
			if df.empty:
				with open(os.path.join(self.test_result_comparison_path, 'test_result_comparison.csv'), 'a', encoding = 'utf-8', newline= '') as f:
					wr = csv.writer(f)
					wr.writerow(['Down Sample Factor', 'Model type', 'Optimizer', 'Initial learning rate', 'Loss function', 'Batch size', 'Best or last', 'Stop Epoch', 'Unet Layer', 'First Layer Kernel Number', 'Test loss'])
					f.close()
		else:
			with open(os.path.join(self.test_result_comparison_path, 'test_result_comparison.csv'), 'a', encoding = 'utf-8', newline= '') as f:
				wr = csv.writer(f)
				wr.writerow(['Down Sample Factor', 'Model type', 'Optimizer', 'Initial learning rate', 'Loss function', 'Batch size', 'Best or last', 'Stop Epoch', 'Unet Layer', 'First Layer Kernel Number', 'Test loss'])
				f.close()

        # Always fill in the test result for this trial.
		with open(os.path.join(self.test_result_comparison_path, 'test_result_comparison.csv'), 'a', encoding = 'utf-8', newline= '') as f:
			wr = csv.writer(f)
			wr.writerow([self.down_factor, self.model_type, self.optimizer_choice, self.initial_lr, self.loss_function_name, self.batch_size, which_unet, stop_epoch, self.UnetLayer, self.first_layer_numKernel, '%.6f' % test_epoch_loss])
			f.close()