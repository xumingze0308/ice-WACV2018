# Multi-Task Spatiotemporal Neural Networks for Structured Surface Reconstruction

## Introduction

This is a PyTorch implementation for our WACV 2018 paper "[`Multi-Task Spatiotemporal Neural Networks for Structured Surface Reconstruction`](https://arxiv.org/pdf/1801.03986.pdf)".

![Alt Text](demo/Movie_20140401_03_033.gif)

***Note:*** The pretrained models are trained on the [`split1`](./data/ice.json) of following larger dataset.

## Environment

- The code is developed with CUDA 9.0, ***Python >= 3.6***, ***PyTorch >= 1.0***

## Data Preparation

1. Download the raw data at `ftp://data.cresis.ku.edu/data/rds/2014_Greenland_P3/CSARP_music3D/`

    - If you don't want to preprocess the raw data by yourself, please use [`create_slices.m`](./scripts/create_slices_64x64/create_slices.m) to generate radar images and [`convert_mat_to_npy.py`](./scripts/convert_mat_to_npy.py) to convert them from MATLAB to NumPy files.

2. If you want to use our [dataloaders](./lib/datasets), please make sure to put the files as the following structure:
    ```
    $YOUR_PATH_TO_CRESIS_DATASET
    ├── slices_mat_64x64/
    |   ├── 20140325_05/
    |   |   ├── 001/
    |   |   |   ├── 00001.mat
    |   |   |   ├── ...
    |   |   ├── ...
    │   ├── ...
    |
    ├── slices_npy_64x64/
    |   ├── 20140325_05/
    |   |   ├── 001/
    |   |   |   ├── 00001.npy
    |   |   |   ├── ...
    |   |   ├── ...
    |   ├── ...
    ```

3. Create softlinks of datasets:
    ```
    cd ice-WACV2018
    ln -s $YOUR_PATH_TO_CRESIS_DATASET data/CReSIS
    ln -s data/target data/CReSIS/target
    ```

## Pretrained Models

- Download the pretrained models at [`model_zoo`](./model_zoo).

## Training

- C3D
```
cd ice-WACV2018
# Default Hyperparameters
python tools/c3d/train.py
# OR
python tools/c3d/train.py --gpu $CUDA_VISIBLE_DEVICES --batch_size $BS --lr $LR
```

- Extract C3D Features
```
cd ice-WACV2018
# Default Hyperparameters
python tools/c3d/extract_features.py
# OR
python tools/c3d/extract_features.py --gpu $CUDA_VISIBLE_DEVICES --batch_size $BS --checkpoint $C3D_CHECKPOINT
```

- RNN
```
cd ice-WACV2018
# Default Hyperparameters
python tools/rnn/train.py
# OR
python tools/rnn/train.py --gpu $CUDA_VISIBLE_DEVICES --batch_size $BS --lr $LR
```

## Evaluation
```
cd ice-WACV2018
# Default Hyperparameters
python demo/e2e_eval.py
# OR
python demo/e2e_eval.py --gpu $CUDA_VISIBLE_DEVICES --batch_size $BS --c3d_pth $C3D_CHECKPOINT --rnn_pth $RNN_CHECKPOINT
```

## Citations

If you are using the data/code/model provided here in a publication, please cite our papers:

    @inproceedings{icesurface2018wacv,
        title = {Multi-Task Spatiotemporal Neural Networks for Structured Surface Reconstruction},
        author = {Mingze Xu and Chenyou Fan and John D. Paden and Geoffrey C. Fox and David J. Crandall},
        booktitle = {IEEE Winter Conference on Applications of Computer Vision (WACV)},
        year = {2018}
    }

    @inproceedings{icesurface2017icip, 
        title = {Automatic estimation of ice bottom surfaces from radar imagery},
        author = {Mingze Xu and David J. Crandall and Geoffrey C. Fox and John D. Paden},
        booktitle = {IEEE International Conference on Image Processing (ICIP)},
        year = {2017}
    }
