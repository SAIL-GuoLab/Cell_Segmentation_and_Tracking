## Cell_Segmentation_and_Tracking/Cell_Segmentation/Prediction_only
*By Nanyan "Rosalie" Zhu and Chen "Raphael" Liu, 2019*

This folder aims to provide users with off-the-shelf (or more technically speaking, pre-trained) deep learning models, so that they can apply these models on their own data to generate segmentation predictions with little effort.

## Folder Hierarchy
```
Prediction_only
└── Step01_raw_channels
└── Step02_channel_combined_input
└── deep_learning_model
└── model_weights
└── results
└── Image_preprocessing.py
└── Visualization.py
└── misc_functions.py
```

The **Prediction_only** folder contains three files, <u>Image_preprocessing.py</u>, <u>Visualization.py</u>, and <u>misc_functions.py</u>, along with five subfolders, **Step01_raw_channels**, **Step02_channel_combined_input**, **deep_learning_model**, **model_weights** and **results**.

<u>Image_preprocessing.py</u> is a quick python file for image preprocessing. It takes in the raw input channels as three pseudo-color channels (R: Nucleus, G: Cytoplasm, B: DIC) and, after a series of image processing steps, generates a channel-concatenated RGB output. The raw input channels, each as individual ".tif" images, shall be placed in the subfolders under the folder **Step01_raw_channels**. An example is shown in **Step01_raw_channels/A1**. The subfolder **A1** stands for the name of the well in which the cells were raised. The three raw input images, <u>A1_w1594_T01.tif</u>, <u>A1_w4Cy5_T01.tif</u> and <u>A1_DIC_T01.tif</u>, respectively stands for the Nucleus, Cytoplasm and DIC channel.

 
