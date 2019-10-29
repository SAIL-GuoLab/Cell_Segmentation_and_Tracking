#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Oct 26 11:33:35 2019

@author: Rosalie and Raphael
"""

import sys
import os
import warnings
from torch.backends import cudnn
import numpy as np
from glob import glob
import matplotlib.pyplot as plt
import nibabel as nib
import scipy.ndimage.morphology as ndi_morph
import skimage.morphology as skimage_morph
from skimage.feature import blob_log
from skimage.measure import label, regionprops
from sklearn.preprocessing import binarize
import scipy.ndimage as ndi

# set the prediction path
os.chdir("/media/sail/HHD8T/DL-Cyto/Danino2019/03_manual_annotation_randomTraining/Cell_Segmentation_and_Tracking/Cell_segmentation/Prediction_only/")
dataset_path = './Step02_channel_combined_input/demo/'
deep_learning_model_path = './deep_learning_model/'
sys.path.append(deep_learning_model_path)

from solver import Solver
from data_loader import get_loader
from misc import printProgressBar

img_predict = glob(dataset_path + '*.png')

warnings.filterwarnings(action = 'once')
# additional parameters for testing
class config():   
    # set options here
    model_type = 'ResAttU_Net'#U_Net/ResAttU_Net
    EE_option = True #True/False
    cuda_idx = int(0)
    num_workers = int(1)
    
    # things you don't need to change
    img_prediction = img_predict
    result_path = './result/%s_EE%s/' % (model_type, EE_option)
    model_weights_path = './model_weights/%s_EE%s.pkl' % (model_type, EE_option)
    patch_num = 24
    row_num = 4
    patch_size = [800, 800]
    col_num = 6
    current_prediction_path = result_path + 'predictions/'

cudnn.benchmark = True
if config.model_type not in ['U_Net','ResAttU_Net']:
    print('ERROR!! model_type should be selected in U_Net/ResAttU_Net')
    print('Your input for model_type was %s'%config.model_type)
#%%

# initialize the test loader
prediction_loader = get_loader(config, mode = 'prediction')

# test model
solver = Solver(config, prediction_loader)
solver.prediction()

#%% paste patches
# visualize the prediction
img = config.img_prediction
prediction = list(np.sort(glob(config.current_prediction_path + '*.npy' )))

# recover patches
whole_img_predic_path = config.current_prediction_path + 'prediction_whole/'
if not os.path.exists(whole_img_predic_path):
    os.mkdir(whole_img_predic_path)

patch_num = 24
row_num = 4
patch_size = [800, 800]
col_num = 6


verline = 4792 // col_num 

ver_margin = int(float(((patch_size[1] - verline) - 4792 % verline ) / (col_num - 1)))

ver_padding = patch_size[1] - (verline - ver_margin)
verline = verline - ver_margin

horiline = 3200 // row_num
hori_margin = int(float(((patch_size[0] - horiline) - 3200 % horiline) / (row_num - 1)))
hori_padding = patch_size[0] - (horiline - hori_margin)
horiline = horiline - hori_margin

# for index in range(len(prediction)):
for index in range(patch_num * len(img_predict)):
    
    if index % patch_num == 0:
        prediction_img = np.zeros([3200, 4792])


    prediction_img[horiline * (index % patch_num % row_num):(horiline * (index % patch_num % row_num + 1) + hori_padding), \
                   verline * (index % patch_num // row_num):(verline * (index % patch_num //row_num + 1) + ver_padding)] = np.load(prediction[index])
    filename = prediction[index].split('/')[-1][-30:-23]
    if index % patch_num == patch_num - 1:
        prediction_img = binarize(prediction_img, threshold = 0.2)
        #plt.imshow(prediction_img, cmap = 'gray')
        #print(filename)
        np.save(whole_img_predic_path + filename + '.npy', prediction_img)
whole_img_predic_list = list(np.sort(glob(whole_img_predic_path + '*.npy')))

#%%
# change the imgage you would like to look at by modifing idx below
idx = 0
predicion_filename_list = [prediction.split('/')[-1].split('.')[0] for prediction in whole_img_predic_list] 
prediction_idx = predicion_filename_list.index(config.img_prediction[idx].split('/')[-1].split('.')[0].replace('Combined_', '') )
sample_prediction_path = whole_img_predic_list[prediction_idx]
sample_prediction = np.load(sample_prediction_path)
sample_prediction_upsampled = np.float32(sample_prediction)
sample_prediction_3d = np.repeat(sample_prediction_upsampled[:, :, np.newaxis], 3, axis = 2)
sample_img = plt.imread(config.img_prediction[idx])[:,:,0:3]
# show a sample of segmentation result
plt.rcParams['figure.figsize'] = [20, 10]
plt.imshow(sample_prediction_3d * 0.5 + sample_img * 0.5)
plt.title(config.img_prediction[idx].split('/')[-1].split('.')[0].replace('Combined_', 'Mask') )
#%% postprocessing (read this in our paper)
# save postprocessed png mask and post postprocessed nifti files with 30 frames
warnings.filterwarnings(action = 'ignore')
# version is our suffix to label different models and approches
version = 'ResAttUnet_edge'
well_list = ['A1']
for well_idx in range(len(well_list)):
    well = well_list[well_idx]
    print('Start Processing: well-' + well)
    sub_prediction_list = []
    sub_img_list = []
    for file in whole_img_predic_list:
        if well in file.split('/')[-1]:
            sub_prediction_list.append(file)
    for file in img_predict:
        if well in file.split('/')[-1]:
            sub_img_list.append(file)
     
    # change the image dimensions(image height, image width, number of time points)
    if len(sub_prediction_list) >=30:
        cell_mask_3D = np.float32(np.zeros((3200, 4792, 30)))
        nuclei_mask_3D = np.float32(np.zeros((3200, 4792, 30)))
        nuclei_segmentation_3D = np.float32(np.zeros((3200, 4792, 30)))
        nuclei_channel_cell_segmentation_3D = np.float32(np.zeros((3200, 4792, 30)))
    else:
        print('no 30 frames to be saved in one well')

    for slice_index in range(len(sub_prediction_list)):
        sample_img = plt.imread(sub_img_list[slice_index])
        predict_downsampled = np.load(sub_prediction_list[slice_index])
        nucleus_image = sample_img[:, :, 0].copy()
        cyto_image = sample_img[:, :, 1].copy()

        abs_cyto = 2 * (cyto_image - nucleus_image)
        abs_cyto[abs_cyto > 1] = 1
        abs_cyto[abs_cyto < 0] = 0

        abs_nucleus = nucleus_image - abs_cyto
        abs_nucleus[abs_nucleus > 1] = 1
        abs_nucleus[abs_nucleus < 0] = 0

        preliminary_nuclei_mask = binarize(abs_nucleus, threshold = 0.5)
        preliminary_nuclei_mask = ndi_morph.binary_fill_holes(preliminary_nuclei_mask)

        preliminary_nuclei_mask = ndi.median_filter(preliminary_nuclei_mask, 5)

        blobs_log = blob_log(preliminary_nuclei_mask, min_sigma = 11, max_sigma = 16, num_sigma = 10, threshold = 0.2)
        blobs_log[:, 2] = blobs_log[:, 2] * np.sqrt(2)
        blobs_list = [blobs_log]
        colors = ['red']
        titles = ['Laplacian of Gaussian']

        x_list = blobs_log[:, 0]
        y_list = blobs_log[:, 1]
        r_list = blobs_log[:, 2]

        nuclei_mask = np.zeros((sample_img.shape[0], sample_img.shape[1]))

        for nuclei_index in range(len(y_list)):
            for x_coord in range(int(max(0, x_list[nuclei_index] - r_list[nuclei_index])), int(min(sample_img.shape[0], x_list[nuclei_index] + r_list[nuclei_index]))):
                for y_coord in range(int(max(0, y_list[nuclei_index] - r_list[nuclei_index])), int(min(sample_img.shape[1], y_list[nuclei_index] + r_list[nuclei_index]))):  
                    if (x_coord - x_list[nuclei_index]) ** 2 + \
                    (y_coord - y_list[nuclei_index]) ** 2 <= \
                    (r_list[nuclei_index]) ** 2:
                        nuclei_mask[x_coord, y_coord] = 1
        predict = predict_downsampled
        seed = np.zeros(np.shape(predict))

        for seed_index in range(len(x_list)):
            if predict[int(x_list[seed_index]), int(y_list[seed_index])] == 1:
                seed[int(x_list[seed_index]), int(y_list[seed_index])] = 1

        markers = ndi.label(seed)[0]
        distance = ndi.distance_transform_edt(predict)
        labels = skimage_morph.watershed(- predict, markers, mask = np.ones(np.shape(predict)), watershed_line = True)
        fissure_map = labels > 0

        thickened_fissure_map = ndi.minimum_filter(fissure_map, size = 4)

        cell_segmentation = np.logical_and(predict, thickened_fissure_map)

        cell_label_map = label(cell_segmentation)
        for region in regionprops(cell_label_map):
            keep_region = False
            for pixel in region.coords:
                if seed[pixel[0], pixel[1]] == 1:
                    keep_region = True
            if keep_region == False:
                cell_label_map[cell_label_map == region.label] = 0

        cell_mask = cell_label_map > 0
        postprocessed_path = config.result_path + 'postprocessed_png/'
        if not os.path.exists(postprocessed_path):
            os.mkdir(postprocessed_path)    
        save_folder = postprocessed_path + '%s/' % well
        if not os.path.exists(save_folder):
            os.mkdir(save_folder)
        plt.imsave(save_folder + whole_img_predic_list[slice_index].split('/')[-1].replace('.npy', '_postprocessed.png'), cell_mask)
        try:
            cell_mask_3D[:, :, slice_index - 30] = cell_mask
            nuclei_mask_3D[:, :, slice_index - 30] = nuclei_mask
            nuclei_segmentation_3D[:, :, slice_index - 30] = nucleus_image * nuclei_mask
            nuclei_channel_cell_segmentation_3D[:, :, slice_index - 30] = nucleus_image * cell_mask
            printProgressBar(slice_index, 30)
        except:
            printProgressBar(slice_index, len(sub_prediction_list))

    # Nifti shell is a original nifti scan that we used to keep all the files with the same affine and header
    # You may need to change your nifti shell here
    try:
        NIFTI_shell_file = './Step02_channel_combined_input/demo/1600*1600_nifti_shell.nii.gz'
        # change the result path to where you want to save this file
        result_path = config.result_path + 'nifti_30frames/'
        if not os.path.exists(result_path):
            os.mkdir(result_path)
        NIFTI_shell = nib.load(NIFTI_shell_file)
        cell_mask_3D_patch_nii = nib.Nifti1Image(cell_mask_3D, NIFTI_shell.affine)
        nuclei_mask_3D_patch_nii = nib.Nifti1Image(nuclei_mask_3D, NIFTI_shell.affine)
        nuclei_segmentation_3D_patch_nii = nib.Nifti1Image(nuclei_segmentation_3D, NIFTI_shell.affine)
        nuclei_channel_cell_segmentation_3D_patch_nii = nib.Nifti1Image(nuclei_channel_cell_segmentation_3D, NIFTI_shell.affine)
    
        nib.save(cell_mask_3D_patch_nii, result_path + 'cell_mask_3D_%s_%s.nii.gz' % (well, version))
        nib.save(nuclei_mask_3D_patch_nii, result_path + 'nuclei_mask_3D_%s_%s.nii.gz' % (well, version))
        nib.save(nuclei_segmentation_3D_patch_nii, result_path + 'nuclei_segmentation_3D_%s_%s.nii.gz' % (well, version))
        nib.save(nuclei_channel_cell_segmentation_3D_patch_nii, result_path + 'nuclei_channel_cell_segmentation_3D_%s_%s.nii.gz' % (well, version))
    except:
        print('cannot save nifti files')
