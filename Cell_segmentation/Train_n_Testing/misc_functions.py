#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Oct 26 21:41:46 2019

@author: Rosalie and Raphael
"""

def find_sub_list(folder, img_type):
    # img_type = 'cyto'/ 'nucleus'/ 'DIC'/ 'Turq'/ 'Cit'
    # folder = A1 - D5 + 'bg'
    folder_list = []
    for i in ['A', 'B', 'C', 'D'] :
        for j in range(1,7):
            if i + str(j) == 'D6':
                folder_list.append('bg')
            else:
                folder_list.append(i + str(j))
    folder_idx = folder_list.index(folder)
    img_files = All_img_files[folder_idx]
    name_list = ['cyto', 'nucleus', 'DIC', 'Turq', 'Cit']
    img_type_list = ['594_', 'Cy5_', 'DIC_', 'Turq_', 'Cit_']
    img_type_files = [[] for i in range(len(img_type_list))]

    for index in range(len(img_files)):
        for img in img_type_list:
            if img in img_files[index]:
                img_type_files[img_type_list.index(img)].append(img_files[index])
    img_sublist = img_type_files[name_list.index(img_type)]
    return img_sublist

def rescaling(img, lower_fraction, upper_fraction, bins = 65536, out_range = (0, 2**16 - 1)):  
    dtype = img.dtype.type
    
    img_cdf, bins = exposure.cumulative_distribution(img, bins)
    
    if lower_fraction == None:
        min_threshold = bins[0]
    else:
        min_idx = np.where(img_cdf < lower_fraction)[0][-1]
        min_threshold = bins[min_idx]
    if upper_fraction == None:
        max_threshold = bins[-1]
    else:
        max_idx = np.where(img_cdf > upper_fraction)[0][0]
        max_threshold = bins[max_idx]
    
    better_contrast = exposure.rescale_intensity(img, in_range = (min_threshold, max_threshold), out_range = out_range)
    
    return np.array(better_contrast, dtype = dtype)

def plot_img_and_hist(image, axes, bins=1024, xlim = True):
    """Plot an image along with its histogram and cumulative histogram.
​
    """
    image = img_as_float(image)
    ax_img, ax_hist = axes
    ax_cdf = ax_hist.twinx()
​
    # Display image
    ax_img.imshow(image, cmap=plt.cm.gray)
    ax_img.set_axis_off()
​
    # Display histogram
    ax_hist.hist(image.ravel(), bins=bins, color='black')
    ax_hist.ticklabel_format(axis='y', style='scientific', scilimits=(0, 0))
    ax_hist.set_xlabel('Pixel intensity')
    if xlim == False :
        ax_hist.set_xticks([])
    else:
        ax_hist.set_xlim(0, 1)
​
    # Display cumulative distribution
    img_cdf, bins = exposure.cumulative_distribution(image, bins)
    ax_cdf.plot(bins, img_cdf, 'r')
    ax_cdf.set_yticks([])
​
    return ax_img, ax_hist, ax_cdf

def plot_img(img_path, img_num = 4, title = None):
    lyout = [int(np.sqrt(img_num))] * 2
    for i in range(lyout[0]):
        for j in range(lyout[1]):
            idx = i * lyout[1] + j 
            plt.subplot(lyout[0], lyout[1], idx + 1)
            img = plt.imread(img_path[idx]);
            plt.imshow(img);
            if title != None:
                plt.title(title[idx])
            plt.axis('off');
            
def printProgressBar (iteration, total, prefix = '', suffix = '', decimals = 1, length = 100, fill = '█'):
    """
    Call in a loop to create terminal progress bar
    @params:
        iteration   - Required  : current iteration (Int)
        total       - Required  : total iterations (Int)
        prefix      - Optional  : prefix string (Str)
        suffix      - Optional  : suffix string (Str)
        decimals    - Optional  : positive number of decimals in percent complete (Int)
        length      - Optional  : character length of bar (Int)
        fill        - Optional  : bar fill character (Str)
    """
    percent = ("{0:." + str(decimals) + "f}").format(100 * (iteration / float(total)))
    filledLength = int(length * iteration // total)
    bar = fill * filledLength + '-' * (length - filledLength)
    print('\r%s |%s| %s%% %s' % (prefix, bar, percent, suffix), end = '')
    # Print New Line on Complete
    if iteration + 1 == total: 
        print('\r%s |%s| %s%% %s' % (prefix, fill * length, 100, suffix), end = '')