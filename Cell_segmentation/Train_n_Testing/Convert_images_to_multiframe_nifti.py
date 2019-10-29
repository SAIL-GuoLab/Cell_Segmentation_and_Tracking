#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Oct 26 22:16:19 2019

@author: Rosalie and Raphael
"""
import os
import numpy as np
import nibabel as nib
from glob import glob

import matplotlib.pyplot as plt


dataset_path = './raw/'
img_folder_path = list(np.sort(glob(dataset_path + '*/')))
All_img_files = []
for img_folder in img_folder_path:
    All_img_files.append(list(np.sort(glob(img_folder + '*.tif'))))
    
    
folder_list = []
for i in ['A', 'B', 'C', 'D'] :
    for j in range(1,7):
        if i + str(j) == 'D6':
            pass
        else:
            folder_list.append(i + str(j))
        
for folder in folder_list:    
    Combined_path = dataset_path + folder + '/Combined3Channels/'
    img_list = list(np.sort(glob(Combined_path + '*')))
    result_path = './All_well_prediction/%s_Prediction/' % folder
    if not os.path.exists(result_path):
        os.mkdir(result_path)
    NIFTI_shell_file = '/Nifti_shell/20190621_111240_generate_nifti_1_1/strct/1600*1600.nii.gz' # the nifti shell you want to use
    NIFTI_shell = nib.load(NIFTI_shell_file)
    img_RGB_4D = np.float32(np.zeros((3200, 4792, 30, 3)))
    for slice_idx in range(30):
        img = plt.imread(img_list[slice_idx])
        img_RGB_4D[:, :, slice_idx, :] = img[:, :, 0:3] * 255
    rgb_dtype = np.dtype([('R', 'uint8'), ('G', 'uint8'), ('B', 'uint8')])
    shape_3d = img_RGB_4D.shape[0:3]
    img_RGB_hyper3D = img_RGB_4D.copy().astype(np.uint8).view(dtype = rgb_dtype).reshape(shape_3d)
    img_RGB_nifti = nib.Nifti1Image(img_RGB_hyper3D, NIFTI_shell.affine)
    nib.save(img_RGB_nifti, result_path + '%s_RGB_images_30_frames.nii.gz' % folder)