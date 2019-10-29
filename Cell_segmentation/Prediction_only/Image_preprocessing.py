#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Oct 26 12:19:50 2019

@author: Rosalie and Raphael
"""
import os
os.chdir("/media/sail/HHD8T/DL-Cyto/Danino2019/03_manual_annotation_randomTraining/Cell_Segmentation_and_Tracking/Cell_segmentation/Prediction_only/")

import numpy as np
from glob import glob

import matplotlib.pyplot as plt

import scipy.ndimage as ndi

import skimage.exposure as exposure

from misc_functions import plot_img, rescaling, find_sub_list, printProgressBar

# dataset path is the raw data path that contains images for 3 channels (in our case DIC, Neucleus and Cytoplasm)
dataset_path = './raw/'
img_folder_path = list(np.sort(glob(dataset_path + '*/')))
All_img_files = []
for img_folder in img_folder_path:
    All_img_files.append(list(np.sort(glob(img_folder + '*.tif'))))

# create folder list
folder_list = ['A1']
# change your folder list here (in our case A1-D5 stand for different wells)
# =============================================================================
# for i in ['A', 'B', 'C', 'D'] :
#     for j in range(1,7):
#         if i + str(j) == 'D6':
#             folder_list.append('bg')
#         else:
#             folder_list.append(i + str(j))
# =============================================================================
img_type_list = ['cyto', 'nucleus', 'DIC']

# preprocess 3 channel images
for folder in folder_list:
    Combined_path = dataset_path + folder + '/Combined3Channels/'
    if not os.path.exists(Combined_path):
        os.mkdir(Combined_path)
    DIC_sublist = find_sub_list(folder, 'DIC', All_img_files)
    Cyto_sublist = find_sub_list(folder, 'cyto', All_img_files)
    Nucleus_sublist = find_sub_list(folder, 'nucleus', All_img_files)
    print('Start Processing: ' + folder)
    
    for time_idx in range(len(DIC_sublist)):
        # these filter size and scaling range may vary depending on your data
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
        DIC_corrected_img = DIC_img/DIC_filtered
        DIC_range_corrected = np.uint16(exposure.rescale_intensity(DIC_corrected_img, out_range = (0, 2**16-1)))
        DIC_filtered_rescaled = rescaling(DIC_range_corrected, 0.001, 0.999)

        # Combined
        combined = np.concatenate((nucleus_filtered_rescaled[..., np.newaxis], cyto_filtered_rescaled[..., np.newaxis], DIC_filtered_rescaled[..., np.newaxis]), axis = -1) / (2 ** 16 - 1) 
        
        printProgressBar(time_idx, len(DIC_sublist))
        fname = DIC_sublist[time_idx][-7:] ##change your filename here
        plt.imsave(Combined_path + folder + '_Combined_'+ fname + '.png', combined, vmin = 0.0, vmax = 2 ** 16 - 1)
        
        
#Check Combined Images
Combined_path = dataset_path + 'A1' + '/Combined3Channels/'
Combined_sublist = list(np.sort(glob(Combined_path+ '/*')))
title_list = [x[-19:-4] for x in Combined_sublist]
plt.rcParams['figure.figsize'] = [10,10]
# plot the first combined image
plt.imshow(plt.imread(Combined_sublist[0]))
# plot multiple combined image
#plot_img(Combined_sublist[0], img_num = 1, title = title_list[0])
