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

The **Prediction_only** folder contains three files, __Image_preprocessing.py__, __Visualization.py__, and _misc_functions.py_, along with five subfolders, **Step01_raw_channels**, **Step02_channel_combined_input**, **deep_learning_model**, **model_weights** and **results**.

In **Prediction_only**, users can directly apply the off-the-shelf (or more technically speaking, pre-trained) deep learning models on their own data to generate segmentation predictions. 

In **Train_n_Testing**, users can train their own deep learning model and, if done properly, obtain a version best suits their own data.
