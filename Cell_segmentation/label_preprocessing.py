#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Oct 26 11:20:25 2019

@author: Rosalie and Raphael
"""
## We used 3d slicer for cell segmentation manual annotation, so this file is to convert
## annotated nifti with 6 subfield to a whole png image format.
# import packages
import numpy as np
import nibabel as nib
import matplotlib.pyplot as plt
import os
from glob import glob

# setup paths and check your annotation files
annotation_path = './annotation/'
annotation_list = glob(annotation_path + '*.nii.gz')
print(annotation_list)

# recover patches and convert to png format
def recoverPatches(images, patch_num = 6, plot_img = False):    
    if patch_num == 6:
        row_num = 2
        patch_size = [1600, 1600]
        col_num = patch_num // row_num
    elif patch_num == 12:
        row_num = 4
        patch_size = [816, 1600]
        col_num = patch_num // row_num

    elif patch_num == 24:
        row_num = 4
        patch_size = [816, 816]
        col_num = patch_num // row_num

    verline = 4792 // col_num 

    ver_margin = ((patch_size[1] - verline) - 4792 % verline ) // (col_num - 1)

    ver_padding = patch_size[1] - (verline - ver_margin)
    verline = verline - ver_margin

    horiline = 3200 // row_num
    hori_margin = ((patch_size[0] - horiline) - 3200 % horiline) // (row_num - 1)
    hori_padding = patch_size[0] - (horiline - hori_margin)
    horiline = horiline - hori_margin

    # for index in range(len(prediction)):
    for index in range(patch_num):

        if index % patch_num == 0:
            prediction_img = np.zeros([3200, 4792])


        prediction_img[horiline * (index % patch_num % row_num) :horiline * (index % patch_num % row_num + 1) + hori_padding, \
                       verline * (index % patch_num // row_num) :verline * (index % patch_num //\
                                                                           row_num + 1) + ver_padding] = images[:,:,index]
        
    if plot_img == True:
        plt.imshow(prediction_img, cmap = 'gray')
        plt.axis('off')
    return prediction_img

for path in annotation_list:
    scan = nib.load(path).get_fdata()[:,:,1:-1]
    whole_img = recoverPatches(scan)
    img_name = path.split('/')[-1].split('.')[0] + '.png'
    plt.imsave(annotation_path + img_name, whole_img, vmin = 0.0, vmax = 1, cmap = 'gray', format = 'png')