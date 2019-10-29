import numpy as np
from torch.utils import data
from matplotlib import pyplot as plt

class ImageFolder(data.Dataset):
	def __init__(self, config, mode = 'prediction'):
		"""Initializes image paths and preprocessing module."""
		if mode == 'prediction':
			self.img_list = config.img_prediction
		self.mode = mode
		self.patch_num = config.patch_num
		self.row_num = config.row_num
		self.patch_size = config.patch_size
        
        
		print('image count in {} path :{}'.format(self.mode,len(self.img_list)))

	def __getitem__(self, index):
		"""Reads an image from a file and preprocesses it and returns."""
		patch_num = self.patch_num 
		row_num = self.row_num
		patch_size = np.array(self.patch_size)
		col_num = patch_num // row_num


		verline = 4792 // col_num 

		ver_margin = int(float(((patch_size[1] - verline) - 4792 % verline ) / (col_num - 1)))

		ver_padding = patch_size[1] - (verline - ver_margin)
		verline = verline - ver_margin

		horiline = 3200 // row_num
		hori_margin = int(float(((patch_size[0] - horiline) - 3200 % horiline) // (row_num - 1)))
		hori_padding = patch_size[0] - (horiline - hori_margin)
		horiline = horiline - hori_margin

		img_path = self.img_list[index // self.patch_num]

		patch_id = index % self.patch_num

		img = plt.imread(img_path)[horiline * (patch_id % row_num):horiline * (patch_id % row_num + 1) + hori_padding, verline * (patch_id// row_num): verline * (patch_id // row_num + 1) + ver_padding, 0:3]

        # resize images by down sampling fator
		img_resized = np.float32(img)
            
		if img_resized.shape[0] % 2**4 != 0 or img_resized.shape[1] % 2**4 != 0 :
			print('Figure dimention error')
			print(index, img_resized.shape[0], img_resized.shape[1])
             
		return img_resized

	def __len__(self):
		"""Returns the total number of font files."""
		return len(self.img_list) * self.patch_num

def get_loader(config, mode = 'prediction', shuffle = False):
	"""Builds and returns Dataloader."""
	batch_size = int(1)
	dataset = ImageFolder(config, mode = mode)
	data_loader = data.DataLoader(dataset = dataset,
								  batch_size = batch_size,
								  shuffle = shuffle,
								  num_workers = config.num_workers)
	return data_loader