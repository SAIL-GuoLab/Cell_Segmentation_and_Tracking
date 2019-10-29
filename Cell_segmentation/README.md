# Cell_Segmentation_and_Tracking/Cell_Segmentation
*By Nanyan "Rosalie" Zhu and Chen "Raphael" Liu, 2019*

The ability to extrapolate gene expression dynamics in living single cells requires robust cell segmentation, and one of the challenges is the amorphous or irregularly shaped cell boundaries. To address this issue, we modified the U-Net architecture to segment cells in fluorescence widefield microscopy images and quantitatively evaluated its performance.

## Folder Hierarchy
```
Cell_Segmentation
├── Prediction_only
│   └── Step01_raw_channels
│   └── Step02_channel_combined_input
│   └── deep_learning_model
│   └── model_weights
│   └── results
│   └── Image_preprocessing.py
│   └── Visualization.py
│   └── misc_functions.py
└── Train_n_Testing
    └── deep_learning_model
    └── ...
```

The **Cell_Segmentation** folder contains two subfolders, **Prediction_only** and **Train_n_Testing**.

In **Prediction_only**, users can directly apply the off-the-shelf (or more technically speaking, pre-trained) deep learning models on their own data to generate segmentation predictions. 

In **Train_n_Testing**, users can train their own deep learning model and, if done properly, obtain a version best suits their own data.
