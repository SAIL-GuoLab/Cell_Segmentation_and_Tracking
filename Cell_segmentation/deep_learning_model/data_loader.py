import os
import random
from glob import glob
from random import shuffle
import numpy as np
import torch
from skimage.transform import resize
from torch.utils import data
from torchvision import transforms as T
from torchvision.transforms import functional as F
from matplotlib import pyplot as plt
from sklearn.feature_extraction import image as sklearn_image

class ImageFolder(data.Dataset):
	def __init__(self, config, mode = 'train'):
		"""Initializes image paths and preprocessing module."""
		if mode == 'train':
			self.img_list = config.img_train
			self.GT_list = config.GT_train
		elif mode == 'validation':
			self.img_list = config.img_val
			self.GT_list = config.GT_val
		elif mode == 'test':
			self.img_list = config.img_test
			self.GT_list = config.GT_test
		self.mode = mode
		self.patch_num = config.patch_num // config.down_factor // config.down_factor
		if mode == 'train':
			self.select_num = list(range(self.patch_num))[:-1]
		elif mode == 'validation':
			self.select_num = [list(range(self.patch_num))[-1]]
		elif mode == 'test':
			self.select_num = list(range(self.patch_num))
            
		self.row_num = config.row_num
		self.debug = config.debug
		self.patch_size = config.patch_size
		self.down_factor = config.down_factor
        
        
		print('image count in {} path :{}'.format(self.mode,len(self.img_list)))
		print("GT count in {} path :{}".format(self.mode,len(self.GT_list)))

	def __getitem__(self, index):
		"""Reads an image from a file and preprocesses it and returns."""
		down_factor = self.down_factor
		patch_num = self.patch_num 
		row_num = self.row_num // down_factor
		patch_size = np.array(self.patch_size) * down_factor
		col_num = patch_num // row_num


		verline = 4792 // col_num 

		ver_margin = int(float(((patch_size[1] - verline) - 4792 % verline ) / (col_num - 1)))

		ver_padding = patch_size[1] - (verline - ver_margin)
		verline = verline - ver_margin

		horiline = 3200 // row_num
		hori_margin = int(float(((patch_size[0] - horiline) - 3200 % horiline) // (row_num - 1)))
		hori_padding = patch_size[0] - (horiline - hori_margin)
		horiline = horiline - hori_margin

		img_path = self.img_list[index // len(self.select_num)]
		GT_path = self.GT_list[index // len(self.select_num)]
		
		patch_idx = index % len(self.select_num)
		patch_id = self.select_num[patch_idx]

		img = plt.imread(img_path)[horiline * (patch_id % row_num):horiline * (patch_id % row_num + 1) + hori_padding, verline * (patch_id// row_num): verline * (patch_id // row_num + 1) + ver_padding, 0:3]
		GT = plt.imread(GT_path)[horiline * (patch_id % row_num):horiline * (patch_id % row_num + 1) + hori_padding, verline * (patch_id // row_num): verline * (patch_id // row_num + 1) + ver_padding, 0]

        # resize images by down sampling fator
		if down_factor != 1:
			img_resized = np.float32(resize(img, (img.shape[0] / down_factor, img.shape[1] / down_factor), anti_aliasing=True))
			GT_resized = np.float32(resize(GT, (GT.shape[0] / down_factor, GT.shape[1] / down_factor), anti_aliasing=True))
			GT_resized[GT_resized >= 0.5] = 1; GT_resized[GT_resized < 0.5] = 0; 
		else:
			img_resized = np.float32(img)
			GT_resized = np.float32(GT)
		
		if self.debug == True:
			print('Shuffle Index:', index, ', Patch ID:', index % 6)
			print('Image Path:', img_path)
			print('Input Image Shape:', img_resized.shape, ', Ground Truth Image Shape:', GT_resized.shape)
			print('Original dtype: ', img.dtype, ', Current dtype: ', img_resized.dtype)
            
		if img_resized.shape[0] % 2**3 != 0 or img_resized.shape[1] % 2**3 != 0 or GT_resized.shape[0] % 2**3 != 0 or GT_resized.shape[1] % 2**3 != 0:
			print('Figure dimention error')
			print(self.select_num)
			print(index, img_resized.shape[0], img_resized.shape[1])
             
		return img_resized, GT_resized

	def __len__(self):
		"""Returns the total number of font files."""
		return len(self.img_list) * len(self.select_num)

def get_loader(config, mode = 'train', shuffle = True):
	"""Builds and returns Dataloader."""
	if mode == 'test':
		batch_size = int(1)
	else:
		batch_size = config.batch_size
	dataset = ImageFolder(config, mode = mode)
	data_loader = data.DataLoader(dataset = dataset,
								  batch_size = batch_size,
								  shuffle = shuffle,
								  num_workers = config.num_workers)
	return data_loader