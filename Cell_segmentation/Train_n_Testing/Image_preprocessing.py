#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Oct 26 12:19:50 2019

@author: Rosalie and Raphael
"""

import os
import cv2
import random
import datetime as dt
import numpy as np
import pandas as pd
from glob import glob

import matplotlib.pyplot as plt
import matplotlib.pylab as plb
from matplotlib import colors

from scipy import interpolate
import scipy.ndimage as ndi
import scipy.ndimage.filters as scifilters
import scipy.ndimage.morphology as ndi_morph

import skimage.exposure as exposure
import skimage.feature as feature
import skimage.filters as filters
import skimage.morphology as skimage_morph
import skimage.segmentation as seg
from skimage import measure, data, img_as_float, color, feature

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.modules.utils import _pair, _quadruple
from misc_functions import plot_img, plot_img_and_hist, rescaling, find_sub_list, printProgressBar

# dataset path is the raw data path that contains images for 3 channels (in our case DIC, Neucleus and Cytoplasm)
dataset_path = './raw/'
img_folder_path = list(np.sort(glob(dataset_path + '*/')))
All_img_files = []
for img_folder in img_folder_path:
    All_img_files.append(list(np.sort(glob(img_folder + '*.tif'))))

# create folder list
folder_list = []
for i in ['A', 'B', 'C', 'D'] :
    for j in range(1,7):
        if i + str(j) == 'D6':
            folder_list.append('bg')
        else:
            folder_list.append(i + str(j))
img_type_list = ['cyto', 'nucleus', 'DIC', 'Turq', 'Cit']

# preprocess 3 channel images
for folder in folder_list:
    Combined_path = dataset_path + folder + '/Combined3Channels/'
    if not os.path.exists(Combined_path):
        os.mkdir(Combined_path)
    DIC_sublist = find_sub_list(folder, 'DIC')
    Cyto_sublist = find_sub_list(folder, 'cyto')
    Nucleus_sublist = find_sub_list(folder, 'nucleus')
    print('Start Processing: ' + folder)
    
    for time_idx in range(len(DIC_sublist)):
        # Cytoplasm
        filter_size = 100
        cyto_img = plt.imread(Cyto_sublist[time_idx])
        cyto_scaled = rescaling(cyto_img, 0.01, 0.99)
        cyto_filtered_img = ndi.gaussian_filter(cyto_scaled, filter_size)
        cyto_corrected = cyto_scaled / cyto_filtered_img
        cyto_filtered_rescaled = np.uint16(rescaling(cyto_corrected, 0.1, 0.99))
        
        # Nucleus
        filter_size = 80
        nucleus_img = plt.imread(Nucleus_sublist[time_idx])
        nucleus_scaled = rescaling(nucleus_img, 0.01, None)
        nucleus_filtered_img = ndi.gaussian_filter(nucleus_scaled, filter_size)
        nucleus_corrected = nucleus_scaled / nucleus_filtered_img
        nucleus_filtered_rescaled = np.uint16(rescaling(nucleus_corrected, 0.1, 0.99))

        # DIC 
        DIC_img = plt.imread(DIC_sublist[time_idx])
        DIC_filtered = ndi.gaussian_filter(DIC_img, 25)
        DIC_corrected_img = img/filtered
        DIC_range_corrected = np.uint16(exposure.rescale_intensity(corrected_img, out_range = (0, 2**16-1)))
        DIC_filtered_rescaled = rescaling(range_corrected, 0.001, 0.999)

        # Combined
        combined = np.concatenate((nucleus_filtered_rescaled[..., np.newaxis], cyto_filtered_rescaled[..., np.newaxis], DIC_filtered_rescaled[..., np.newaxis]), axis = -1) / (2 ** 16 - 1) 
        
        printProgressBar(time_idx, len(DIC_sublist))
        plt.imsave(Combined_path + folder + '_Combined_'+ DIC_sublist[time_idx][-7:], combined, vmin = 0.0, vmax = 2 ** 16 - 1, cmap = 'gray', format = 'png')
        
        
#Check Combined Images
Combined_path = dataset_path + 'A1' + '/Combined3Channels/'
Combined_sublist = list(np.sort(glob(Combined_path+ '/*')))
title_list = [x[-19:-4] for x in Combined_sublist]
plt.rcParams['figure.figsize'] = [30,20]
plot_img(Combined_sublist[8:12], img_num = 4, title = title_list[8:12])
