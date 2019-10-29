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
from network import U_Net, ResAttU_Net
from evaluation import CrossEntropy
from misc import printProgressBar
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
	def __init__(self, config, prediction_loader):

		# Data loader
		self.prediction_loader = prediction_loader
		self.prediction_img_list = config.img_prediction
        
		# Models
		self.unet = None
		self.optimizer = None
		self.img_ch = int(3)
		self.output_ch = int(1)
		self.first_layer_numKernel = 32
		self.UnetLayer = 5

		# Hyper-parameters
		self.patch_num = config.patch_num

		# Path
		self.current_prediction_path = config.current_prediction_path
		self.current_model_saving_path = config.model_weights_path

		self.device = torch.device('cuda: %d' % config.cuda_idx)
		self.model_type = config.model_type
		self.build_model()

	def build_model(self):
		"""Build generator and discriminator."""
		if self.model_type =='U_Net':
			self.unet = U_Net(UnetLayer = self.UnetLayer, img_ch = self.img_ch, output_ch = self.output_ch, first_layer_numKernel = self.first_layer_numKernel)
		elif self.model_type == 'ResAttU_Net':
			self.unet = ResAttU_Net(UnetLayer = self.UnetLayer, img_ch = self.img_ch, output_ch = self.output_ch, first_layer_numKernel = self.first_layer_numKernel)

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

	def reset_grad(self):
		"""Zero the gradient buffers."""
		self.unet.zero_grad()

	def prediction(self):
		#======================================= Prediction ====================================#
		#=================================================================================#
		unet_path = self.current_model_saving_path

		self.build_model()
		if not os.path.exists(self.current_prediction_path):
			os.makedirs(self.current_prediction_path)
		self.unet.load_state_dict(torch.load(unet_path))
		self.unet.train(False)
		self.unet.eval()
		prediction_img_list = self.prediction_img_list
		for batch, (img) in enumerate(self.prediction_loader):
			img = img.to(self.device)

			# Reshape the images and GTs to 4-dimensional so that they can get fed to the conv2d layer. (R&R)
			# The new shape has to be (batch_size, num_channels, img_dim1, img_dim2).
			if self.img_ch == 1:
				img = img[:, np.newaxis, :, :]
			else:
				img = img.transpose(1, 3); img = img.transpose(2, 3)

			#plt.subplot(1,2,1); plt.imshow(img[0, 0, :, :].cpu().detach().numpy())

			#  SR : Segmentation Result
			SR = torch.sigmoid(self.unet(img))
        
			# Flatten the prediction and target.
			SR_flat = SR.view(SR.size(0), -1)

			np_img = np.squeeze(SR.cpu().detach().numpy()) 

			# extract filename from test folder
			filename = prediction_img_list[batch//self.patch_num][-19:-4].replace('Combined_', '')

			np.save(self.current_prediction_path + filename + '_' + str(batch % self.patch_num).zfill(2) + '_modelprediction.npy', np_img)
            
			printProgressBar(batch, len(self.prediction_loader))
			del batch, img, SR, SR_flat
			torch.cuda.empty_cache()

		print('done!')
