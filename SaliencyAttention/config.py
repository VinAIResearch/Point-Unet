#!/usr/bin/env python
# -*- coding: utf-8 -*-
# File: config.py

import numpy as np

# unet model
DATA_SAMPLING = 'one_positive' # one_positive, all_positive, random
###
# random: complete random sampling within entire volume.
# one_positive: at least one batch contain tumor label (label > 0)
# all_positive: all batch must contain tumor label (label > 0)
###
MIXUP = False
#Use mixup sample method
RESIDUAL = True
DEPTH = 5
DEEP_SUPERVISION = True
FILTER_GROW = True
INSTANCE_NORM = True
# Use multi-view fusion 3 models for 3 view must be trained
DIRECTION = 'axial' # axial, sagittal, coronal
MULTI_VIEW = False

CA_attention = True
SA_attention = True
# training config
BASE_LR = 0.01

CROSS_VALIDATION = False
CROSS_VALIDATION_PATH = "./10folds.pkl"
FOLD = 2
###
# Use when 5 fold cross validation
# 1. First run generate_5fold.py to save 5fold.pkl
# 2. Set CROSS_VALIDATION to True
# 3. CROSS_VALIDATION_PATH to /path/to/5fold.pkl
# 4. Set FOLD to {0~4}
###
NO_CACHE = True
###
# if NO_CACHE = False, we load pre-processed volume into memory to accelerate training.
# set True when system memory loading is too high
###
TEST_FLIP = True
# Test time augmentation
DYNAMIC_SHAPE_PRED = False
# change PATCH_SIZE in inference if cropped brain region > PATCH_SIZE
ADVANCE_POSTPROCESSING = False
BATCH_SIZE = 1
PATCH_SIZE = [128, 128, 128]
INFERENCE_PATCH_SIZE = [128, 128, 128]

# PATCH_SIZE = [64, 64, 64]
# INFERENCE_PATCH_SIZE = [64, 64, 64]

INTENSITY_NORM = 'modality' # different norm method
STEP_PER_EPOCH = 500
EVAL_EPOCH = 5
MAX_EPOCH = 200

#STEP_PER_EPOCH = 200
#EVAL_EPOCH = 1000
#MAX_EPOCH = 20

#BASEDIR = "/vinai/vuonghn/Research/BraTS/BraTS_data/MICCAI_BraTS_2018_Data_Training" #'/data/dataset/BRATS2018/'
#BASEDIR = ["datasets/BraTS2020/", 'datasets/MICCAI_BraTS_2018_Data_Training/', 'datasets/MICCAI_BraTS_2019_Data_Training/']
BASEDIR = ["../datasets/BraTS2020/", '../datasets/MICCAI_BraTS_2018_Data_Training', '../datasets/MICCAI_BraTS_2019_Data_Training']
#BASEDIR = ["../datasets/BraTS2020/"]
TRAIN_DATASET = 'training'
VAL_DATASET = 'training'   # val or val17 
TEST_DATASET = '.'


BASEDIR = "../datasets/BraTS2020_Validation/"
TRAIN_DATASET = ['val']
VAL_DATASET = 'training'   # val or val17 
TEST_DATASET = 'val'
save_pred = "../save_pred/BraTS2020_val/"

NUM_CLASS = 4
# GT_TEST = False
