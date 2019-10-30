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

The **Prediction_only** folder contains three files, *Image_preprocessing.py*, *Visualization.py*, and *misc_functions.py*, along with five subfolders, **Step01_raw_channels**, **Step02_channel_combined_input**, **deep_learning_model**, **model_weights** and **results**.

- [*Image_preprocessing.py*](https://github.com/SAIL-GuoLab/Cell_Segmentation_and_Tracking/blob/master/Cell_segmentation/Prediction_only/Image_preprocessing.py) is a python file for image preprocessing. It takes in the raw input channel as three pseudo-color channels (R: Nucleus, G: Cytoplasm, B: DIC) and, after a series of image processing steps, generates a channel-concatenated RGB output. The raw input channels, each as individual ".tif" images, shall be placed in the subfolders under the folder **Step01_raw_channels**. An example is shown in **Step01_raw_channels/A1**. The subfolder **A1** stands for the name of the well in which the cells were raised. The three raw input images, *A1_w1594_T01.tif*, *A1_w4Cy5_T01.tif* and *A1_DIC_T01.tif*, respectively stands for the Cytoplasm, Nucleus and DIC channel. The generated channel-combined RGB image, *A1_Combined_T01.png*, is automatically stored in the folder **Step01_raw_channels/A1/Combined3Channels/**.
[]

- [*Visualization.py*](https://github.com/SAIL-GuoLab/Cell_Segmentation_and_Tracking/blob/master/Cell_segmentation/Prediction_only/Visualization.py) is a python file to run the deep learning segmentation.
    - Before running this, the user should have already completed two steps.
        - run [*Image_preprocessing.py*](https://github.com/SAIL-GuoLab/Cell_Segmentation_and_Tracking/blob/master/Cell_segmentation/Prediction_only/Image_preprocessing.py) to generate the channel-combined RGB image(s) (which can be used as a valid input to the deep learning model).
        - migrate (or, more preferrably, copy) the RGB image(s) from the **Step01_raw_channels/*something*/Combined3Channels/** subfolder to the **Step02_channel_combined_input/*project_name*/**. The latter is the path from which [*Visualization.py*](https://github.com/SAIL-GuoLab/Cell_Segmentation_and_Tracking/blob/master/Cell_segmentation/Prediction_only/Visualization.py) will grab the input files to generate segmentation predictions. Please note that you need to modify the parameter "dataset_path" from the default value of **Step02_channel_combined_input/demo/** to the actual one, **Step02_channel_combined_input/*project_name*/**, that you are using.
    - Once the user run this [*Visualization.py*](https://github.com/SAIL-GuoLab/Cell_Segmentation_and_Tracking/blob/master/Cell_segmentation/Prediction_only/Visualization.py), the following things will happen.
        - Segmentation predictions from the deep learning model will be generated and stored under **Prediction_only/result/*DLModelType*/predictions**.
        - Postprocessing (adjacent cell cleaning, etc.) is performed on the segmentation predictions and stored under **Prediction_only/result/*DLModelType*/postprocessed_png/*something*/**.
        - If data in one well exceed 30 time frames, the 3d nifti file (contains 30 time frames in one well) will be generated in **Prediction_only/result/*DLModelType*/nifti_30frames/**.


Important Note: the raw input images need to conform to the size constraints. Currently, they need to be 4792 by 3200 or otherwise the deep learning model cannot work with them.
