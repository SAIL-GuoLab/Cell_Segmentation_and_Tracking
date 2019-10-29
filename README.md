# Cell Segmentation and Tracking
*By Nanyan "Rosalie" Zhu and Chen "Raphael" Liu, 2019*

The purpose of this repository is to design and implement a tool to quantify the cellular behaviors at the single-cell level. We decided to approach this challenge by dissecting it into two stages: cell segmentation and cell tracking. The quantification of cellular properties/behaviors is a natural outcome as the cell tracking stage is complete.

## Repository Hierarchy
```
Cell_Segmentation_and_Tracking
    ├── Cell_Segmentation
    │   ├── Prediction_only
    │   │   └── Step01_raw_channels
    │   │   └── Step02_channel_combined_input
    │   │   └── deep_learning_model
    │   │   └── model_weights
    │   │   └── results
    │   │   └── Image_preprocessing.py
    │   │   └── Visualization.py
    │   │   └── misc_functions.py
    │   └── Train_n_Testing
    │       └── deep_learning_model
    │       └── ...
    └── Cell_Tracking
        └── ...
```

The **Cell_Segmentation_and_Tracking** repository contrains two folders, **Cell_Segmentation** and **Cell_Tracking**. These two folders are very much self-explanatory. Descriptions about their respective contents are described in the documentation files within each of them.

*Note 1: We did not include the "pycache" folders because they can be ignored practically.
*Note 2: We did not include the ".gitattribute" files and "README.md" files in the hierarchy tree. The former are utilized by the git large file storage (git-lfs) while the latter are the documentations, just like the one you are reading right now.
