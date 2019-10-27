#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Oct 26 11:33:35 2019

@author: Rosalie and Raphael
"""

import sys
import os
import warnings
from shutil import copyfile
from torch.backends import cudnn
import random
import numpy as np
from glob import glob
import matplotlib.pyplot as plt
from skimage.morphology import disk
from skimage.filters import rank
from scipy import ndimage
import nibabel as nib
import scipy.ndimage.morphology as ndi_morph
import skimage.morphology as skimage_morph
from skimage.feature import blob_dog, blob_log, blob_doh
from skimage.measure import label, regionprops
from scipy.misc import imresize
from sklearn.preprocessing import binarize
import scipy.ndimage as ndi
from skimage.transform import resize
from skimage.transform import pyramid_expand


dataset_path = './dataset/'
deep_learning_model_path = './deep_learning_model/'
sys.path.append(deep_learning_model_path)

from solver import Solver
from data_loader import get_loader
from evaluation import get_MSE
from misc import printProgressBar

# change your train test folders here
img_train = glob(dataset_path + 'train/*')
GT_train = glob('./annotation/*train.png')
img_val = img_train
GT_val = GT_train
img_test = glob(dataset_path + 'test/*')
GT_test = glob('./annotation/*train.png')

print(len(GT_train), len(img_train), len(GT_val), len(img_val), len(GT_test), len(img_test))


warnings.filterwarnings(action = 'once')
# additional parameters for testing
stop_epoch = None # None stands for best unet [None, 10, 20, 30]
special_note = 'with_val' # some additional notes we would like to use
UnetLayers = int(5) # [int(5), int(4), int(3)]:    
firstlayerkernal = int(32) # [int(64), int(48), int(32), int(16)]:
class config():   
    # model hyper-parameters
    t = int(3) # t for Recurrent step of R2U_Net or R2AttU_Net'
    img_ch = int(3)
    GT_ch = int(1)
    output_ch = int(1)
    num_epochs = int(30)
    num_workers = int(1)
    patch_size = [800, 800]
    patch_num = 24 # the actual patch will be 96 / down_factor ** 2
    row_num = 4
    down_factor = 1 # down sample factor. We used this to speed up our hyperparameters tuning process.
    mode = 'train'
    debug = False

    ############################### Adjust These ###################################
    model_type = 'ResAttU_Net' # 'U_Net/R2U_Net/AttU_Net/R2AttU_Net/ResAttU_Net'
    optimizer_choice = 'SGD'
    annotation_num = len(img_train)
    lr = float(0.05)  # initial learning rate
    loss_function = 'BCE' # BCE/SmoothL1/
    batch_size = int(1)
    UnetLayer = UnetLayers
    withTF = False
    edge_enhance = 'Double' # 'True/Double'
    first_layer_numKernel = firstlayerkernal



    early_stop = False # shall we enable early stop?
    ################################################################################

    if optimizer_choice == 'Adam':
        beta1 = float(0.5) # momentum1 in Adam
        beta2 = float(0.999) # momentum2 in Adam
    elif optimizer_choice == 'SGD':
        momentum = float(0.9)
    else:
        print('No such optimizer available')

    img_train = img_train
    img_val = img_val
    img_test = img_test

    GT_train = GT_train
    GT_val = GT_val
    GT_test = GT_test
    
    result_path = './result/num_annotation_%d/' % annotation_num
    model_weights_path = result_path + 'model_weights/'
    predictions_path = result_path + 'predictions/'
    loss_histories_path = result_path + 'loss_histories/'
    test_result_comparison_path = result_path + 'test_result_comparison/'

    current_model_saving_path = model_weights_path + \
    'edge_enhance_%s/%s/%s/learning_rate_%.4f/loss_function_%s/batch_size_%d/unet_layer_%d/first_layer_numKernel_%d/' % (edge_enhance, model_type, optimizer_choice, lr, loss_function, batch_size, UnetLayer, first_layer_numKernel)
    current_prediction_path = predictions_path + \
    'edge_enhance_%s/%s/%s/learning_rate_%.4f/loss_function_%s/batch_size_%d/unet_layer_%d/first_layer_numKernel_%d/%s/' % (edge_enhance, model_type, optimizer_choice, lr, loss_function, batch_size, UnetLayer, first_layer_numKernel, special_note)
    current_loss_history_path = loss_histories_path + \
    'edge_enhance_%s/%s/%s/learning_rate_%.4f/loss_function_%s/batch_size_%d/unet_layer_%d/first_layer_numKernel_%d/' % (edge_enhance, model_type, optimizer_choice, lr, loss_function, batch_size, UnetLayer, first_layer_numKernel)

    cuda_idx = int(0)


cudnn.benchmark = True
if config.model_type not in ['U_Net','R2U_Net','AttU_Net','R2AttU_Net', 'ResAttU_Net']:
    print('ERROR!! model_type should be selected in U_Net/R2U_Net/AttU_Net/R2AttU_Net/ResAttU_Net')
    print('Your input for model_type was %s'%model_type)
    break

# initialize the test loader
train_loader = get_loader(config,
                          mode = 'train')

validation_loader = get_loader(config,
                               mode = 'validation')
test_loader = get_loader(config,
                         mode = 'test',
                         shuffle = False)

# test model
solver = Solver(config, train_loader, validation_loader, test_loader)
solver.test(which_unet = 'best', stop_epoch = stop_epoch)

# visualize the prediction
img = config.img_test
GT = config.GT_test
try:
    prediction_path = config.current_prediction_path  + 'epoch%d/' % stop_epoch
    prediction = list(np.sort(glob(prediction_path + '*.npy' )))
except:
    print('no stop epoch')
    prediction_path = config.current_prediction_path  + 'best/'
    prediction = list(np.sort(glob(prediction_path + '*.npy')))

# recover patches
whole_img_predic_path = prediction_path + '/prediction_whole/'
if not os.path.exists(whole_img_predic_path):
    os.mkdir(whole_img_predic_path)
down_factor = config.down_factor
patch_num = config.patch_num // config.down_factor // config.down_factor
row_num = config.row_num // down_factor
patch_size = np.array(config.patch_size) * down_factor
col_num = patch_num // row_num


verline = 4792 // col_num 

ver_margin = int(float(((patch_size[1] - verline) - 4792 % verline ) / (col_num - 1)))

ver_padding = patch_size[1] - (verline - ver_margin)
verline = verline - ver_margin

horiline = 3200 // row_num
hori_margin = int(float(((patch_size[0] - horiline) - 3200 % horiline) / (row_num - 1)))
hori_padding = patch_size[0] - (horiline - hori_margin)
horiline = horiline - hori_margin

# for index in range(len(prediction)):
for index in range(patch_num * len(GT_test)):
    
    if index % patch_num == 0:
        prediction_img = np.zeros([3200// down_factor, 4792// down_factor])


    prediction_img[horiline * (index % patch_num % row_num) // down_factor:(horiline * (index % patch_num % row_num + 1) + hori_padding)// down_factor, \
                   verline * (index % patch_num // row_num)// down_factor:(verline * (index % patch_num //\
                                                                                      row_num + 1) + ver_padding)// down_factor] = np.load(prediction[index])
    filename = prediction[index][-34:-23]
    if index % patch_num == patch_num - 1:
        prediction_img = binarize(prediction_img, threshold = 0.2)
        #plt.imshow(prediction_img, cmap = 'gray')
        #print(filename)
        np.save(whole_img_predic_path + filename + '.npy', prediction_img)
whole_img_predic_list = list(np.sort(glob(whole_img_predic_path + '*.npy')))


# change the imgage you would like to look at by modifing idx below
idx = 33
predicion_filename_list = [prediction.split('/')[-1].split('.')[0] for prediction in whole_img_predic_list] 
prediction_idx = predicion_filename_list.index(config.img_test[idx].split('/')[-1].split('.')[0].replace('Combined', 'Mask') )
sample_prediction_path = whole_img_predic_list[prediction_idx]
sample_prediction = np.load(sample_prediction_path)
if config.down_factor != 1:
    sample_prediction_upsampled = np.float32(pyramid_expand(sample_prediction, upscale = config.down_factor, order = 3) > 0.5)
else:
    sample_prediction_upsampled = np.float32(sample_prediction)
sample_prediction_3d = np.repeat(sample_prediction_upsampled[:, :, np.newaxis], 3, axis = 2)
sample_img = plt.imread(config.img_test[idx])[:,:,0:3]
# show a sample of segmentation result
plt.rcParams['figure.figsize'] = [20, 10]
plt.imshow(sample_prediction_3d * 0.5 + sample_img * 0.5)

plt.title(config.img_test[idx].split('/')[-1].split('.')[0].replace('Combined', 'Mask') )


# postprocessing (read this in our paper) and save 3d (the third dimension is time axis) nifti scan
warnings.filterwarnings(action = 'ignore')
# version is our suffix to label different models and approches
version = 'RR_manual_24patch_ResAttUnet_edge'
well_list = ['D4']
for well_idx in range(len(well_list)):
    well = well_list[well_idx]
    print('Start Processing: well-' + well)
    sub_prediction_list = []
    sub_img_list = []
    sub_GT_list = []
    for file in whole_img_predic_list:
        if well in file.split('/')[-1]:
            sub_prediction_list.append(file)
    for file in img_files:
        if well in file.split('/')[-1]:
            sub_img_list.append(file)
    for file in GT_files:
        if well in file.split('/')[-1]:
            sub_GT_list.append(file)
     
    # change the image dimensions(image height, image width, number of time points)
    cell_mask_3D = np.float32(np.zeros((3200, 4792, 30)))
    nuclei_mask_3D = np.float32(np.zeros((3200, 4792, 30)))
    nuclei_segmentation_3D = np.float32(np.zeros((3200, 4792, 30)))
    nuclei_channel_cell_segmentation_3D = np.float32(np.zeros((3200, 4792, 30)))

    for slice_index in range(0, 30):
        sample_img = plt.imread(sub_img_list[slice_index])
        GT = plt.imread(sub_GT_list[slice_index])[:,:,0]
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
        if config.down_factor != 1:
            predict = np.float32(pyramid_expand(predict_downsampled, upscale = config.down_factor, order = 3) > 0.5)
        else:
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

        cell_mask_3D[:, :, slice_index - 30] = cell_mask
        nuclei_mask_3D[:, :, slice_index - 30] = nuclei_mask
        nuclei_segmentation_3D[:, :, slice_index - 30] = nucleus_image * nuclei_mask
        nuclei_channel_cell_segmentation_3D[:, :, slice_index - 30] = nucleus_image * cell_mask
        printProgressBar(slice_index, 30)

    # Nifti shell is a original nifti scan that we used to keep all the files with the same affine and header
    # You need to change your nifti shell here
    NIFTI_shell_file = '/media/sail/SSD1T/Tal_cell_tracking/Nifti/20190621_111240_generate_nifti_1_1/strct/1600*1600.nii.gz'
    # change the result path to where you want to save this file
    result_path = '/media/sail/SSD1T/Tal_cell_tracking/Nifti/%s_Prediction_comparison/' % well
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
