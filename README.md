# Cell Segmentation and Tracking
The purpose of this repository is to design and implement a tool to quantify the cellular behaviors at the single-cell level. We decided to approach this challenge by dissecting it into two stages: cell segmentation and cell tracking. The quantification of cellular properties/behaviors is a natural outcome as the cell tracking stage is complete.

## Repository Hierarchy

Cell_Segmentation_and_Tracking
    ├── Cell_Segmentation
    │   └── Prediction_only
    │       └── Step01_raw_channels
    │       └── Step02_channel_combined_input
    │       └── deep 
    │   └── Train_n_Testing
    └── Cell_Tracking

*Note 1: We did not include the "pycache" folders because they can be ignored practically.
*Note 2: We did not include the ".gitattribute" files and "README.md" files in the hierarchy tree. The former are utilized by the git large file storage (git-lfs) while the latter are the documentations, just like the one you are reading right now.
