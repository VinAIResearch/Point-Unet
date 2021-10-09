#!/usr/bin/env python
# -*- coding: utf-8 -*-
# File: data_loader.py

import numpy as np
import os
from termcolor import colored
from tabulate import tabulate

from tensorpack.utils import logger
from tensorpack.utils.rect import FloatBox
from tensorpack.utils.timer import timed_operation
from tensorpack.utils.argtools import log_once

import random
import pickle
import glob
from tqdm import tqdm
import config
#import cv2
#import skimage
#import nibabel
from utils import crop_brain_region, load_pancreas_img

class BRATS_SEG(object):
    def __init__(self, basedir, mode):
        """
        basedir="/data/dataset/BRATS2018/{mode}/{HGG/LGG}/patient_id/{flair/t1/t1ce/t2/seg}"
        mode: training/val/test
        """
        self.basedir = os.path.join(basedir, mode)
        self.mode = mode
    
    def load_kfold(self):
        with open(config.CROSS_VALIDATION_PATH, 'rb') as f:
            data = pickle.load(f)
        imgs = data["fold{}".format(config.FOLD)][self.mode]
        patient_ids = [x.split("/")[-1] for x in imgs]
        ret = []
        print("Preprocessing {} Data ...".format(self.mode))
        for idx, file_name in tqdm(enumerate(imgs), total=len(imgs)):
            data = {}
            data['image_data'] = {}
            data['file_name'] = file_name
            data['id'] = patient_ids[idx]
            # read modality
            mod = glob.glob(file_name+"/*.nii*")
            assert len(mod) >= 4  # 4mod +1gt
            for m in mod:
                if 'seg' in m:
                    data['gt'] = m
                else:
                    _m = m.split("/")[-1].split(".")[0].split("_")[-1]
                    data['image_data'][_m] = m
            if 'gt' in data:
                if not config.NO_CACHE:
                    data['preprocessed'] = crop_brain_region(data['image_data'], data['gt'])
                    del data['image_data']
                    del data['gt']
            ret.append(data)
        return ret

    def load_3d(self):
        """
        dataset_mode: HGG/LGG/ALL
        return list(dict[patient_id][modality] = filename.nii.gz)
        """
        print("Data Folder: ", self.basedir)

        modalities = ['flair', 't1ce', 't1.', 't2']
        
        if 'training' in self.basedir:
            img_HGG = glob.glob(self.basedir+"/HGG/*")
            img_LGG = glob.glob(self.basedir+"/LGG/*")
            imgs = img_HGG + img_LGG
        else:
            imgs = glob.glob(self.basedir+"/*")
        imgs = [x for x in imgs if 'survival_evaluation.csv' not in x]
        #imgs = imgs[:30]
        # imgs = ["/vinai/vuonghn/Research/BraTS/BraTS_data/MICCAI_BraTS2020_ValidationData/val/BraTS20_Validation_069"]

        
        patient_ids = [x.split("/")[-1] for x in imgs]

        ret = []
        print("Preprocessing Data ...")
        for idx, file_name in tqdm(enumerate(imgs), total=len(imgs)):
            data = {}
            data['image_data'] = {}
            data['file_name'] = file_name
            data['id'] = patient_ids[idx]
            # read modality
            mod = glob.glob(file_name+"/*.nii*")
            # print("file_name  : ", file_name)
            # print("\npatient_ids[idx] ",patient_ids[idx] ,"\nfile_name ",file_name ,"\nmod ",mod)
            
            assert len(mod) >= 4, '{}'.format(file_name)  # 4mod +1gt        
            for m in mod:
                if 'seg' in m:
                    data['gt'] = m
                else:
                    _m = m.split("/")[-1].split(".")[0].split("_")[-1]
                    data['image_data'][_m] = m
            # print("data['image_data'] ",data['image_data'])
            # print("self.basedir ", self.basedir)

            if 'gt' in data:
                if not config.NO_CACHE:
                    data['preprocessed'] = crop_brain_region(data['image_data'], data['gt'])
                    del data['image_data']
                    del data['gt']
                # if config.GT_TEST:
                #     data['preprocessed'] = crop_brain_region(data['image_data'], None, with_gt=False)
                #     del data['image_data']
            else:
                if not config.NO_CACHE:
                    data['preprocessed'] = crop_brain_region(data['image_data'], None, with_gt=False)
                    del data['image_data']


            # if 'gt' in data:
            #     print("Load data from  : ", file_name)
            #     data['preprocessed'] = crop_brain_region(data['image_data'], data['gt'])
            #     del data['image_data']
            #     del data['gt'] 

            #     # if not config.NO_CACHE and not 'training' in self.basedir:
            #     #     data['preprocessed'] = crop_brain_region(data['image_data'], data['gt'])
            #     #     del data['image_data']
            #     #     del data['gt'] 
            # else:

            #     data['preprocessed'] = crop_brain_region(data['image_data'], None, with_gt=False)
            #     del data['image_data']

            ret.append(data)
        
        return ret

    @staticmethod
    def load_from_file(basedir, names):
        brats = BRATS_SEG(basedir, names)
        return  brats.load_kfold()

    @staticmethod
    def load_many(basedir,names, add_gt=True, add_mask=False):
        """
        Load and merges several instance files together.
        """
        if not isinstance(names, (list, tuple)):
            names = [names]
        ret = []
        for n in names:
            brats = BRATS_SEG(basedir, n)
            ret.extend(brats.load_3d())
        return ret

# if __name__ == '__main__':
#     print("Runing")
#     brats2018 = BRATS_SEG("/vinai/vuonghn/Research/BraTS/BraTS_data/MICCAI_BraTS_2018_Data_Training/", "training")
#     brats2018 = brats2018.load_3d()
#     print(len(brats2018))
#     print(brats2018[0])

###############LOG
#     Runing                                                                                                                │                                                                                                                     
# Data Folder:  /vinai/vuonghn/Research/BraTS/BraTS_data/MICCAI_BraTS_2018_Data_Training/training                       │                                                                                                                     
# Preprocessing Data ...                                                                                                │                                                                                                                     
# 100%|███████████████████████████████████████████████████████████████████████████████| 285/285 [00:02<00:00, 91.72it/s]│                                                                                                                     
# 285                                                                                                                   │                                                                                                                     
# {'image_data': {'t1ce': '/vinai/vuonghn/Research/BraTS/BraTS_data/MICCAI_BraTS_2018_Data_Training/training/HGG/Brats18│                                                                                                                     
# _CBICA_AOP_1/Brats18_CBICA_AOP_1_t1ce.nii.gz', 't2': '/vinai/vuonghn/Research/BraTS/BraTS_data/MICCAI_BraTS_2018_Data_│                                                                                                                     
# Training/training/HGG/Brats18_CBICA_AOP_1/Brats18_CBICA_AOP_1_t2.nii.gz', 't1': '/vinai/vuonghn/Research/BraTS/BraTS_d│                                                                                                                     
# ata/MICCAI_BraTS_2018_Data_Training/training/HGG/Brats18_CBICA_AOP_1/Brats18_CBICA_AOP_1_t1.nii.gz', 'flair': '/vinai/│                                                                                                                     
# vuonghn/Research/BraTS/BraTS_data/MICCAI_BraTS_2018_Data_Training/training/HGG/Brats18_CBICA_AOP_1/Brats18_CBICA_AOP_1│                                                                                                                     
# _flair.nii.gz'}, 'file_name': '/vinai/vuonghn/Research/BraTS/BraTS_data/MICCAI_BraTS_2018_Data_Training/training/HGG/B│                                                                                                                     
# rats18_CBICA_AOP_1', 'id': 'Brats18_CBICA_AOP_1', 'gt': '/vinai/vuonghn/Research/BraTS/BraTS_data/MICCAI_BraTS_2018_Da│                                                                                                                     
# ta_Training/training/HGG/Brats18_CBICA_AOP_1/Brats18_CBICA_AOP_1_seg.nii.gz'}  

class PANCREAS_SEG(object):
    def __init__(self, basedir, mode):
        """
        basedir="/data/dataset/BRATS2018/{mode}/{HGG/LGG}/patient_id/{flair/t1/t1ce/t2/seg}"
        mode: training/val/test
        """
        self.basedir = basedir
        self.mode = mode

    def load_3d(self):
        """
        dataset_mode: HGG/LGG/ALL
        return list(dict[patient_id][modality] = filename.nii.gz)
        """
        print("Data Folder: ", self.basedir)
        
        with open(self.mode, 'r') as file:
            imgs = [os.path.join(self.basedir, 'ct', line[:-1]) for line in file]
        
        # imgs = [x for x in imgs if 'survival_evaluation.csv' not in x]
        
        patient_ids = [x[-11:-7] for x in imgs]

        ret = []
        print("Preprocessing Data ...")
        for idx, file_name in tqdm(enumerate(imgs), total=len(imgs)):
            data = {}
            data['image_data'] = file_name
            data['file_name'] = file_name
            data['id'] = patient_ids[idx]
            data['gt'] = file_name.replace('PANCREAS_', 'label').replace('ct', 'seg')      

            if not config.NO_CACHE:
                data['preprocessed'] = load_pancreas_img(data['image_data'], data['gt'])
                del data['image_data']
                del data['gt']

            ret.append(data)
        
        return ret

    @staticmethod
    def load_many(basedir,names, add_gt=True, add_mask=False):
        """
        Load and merges several instance files together.
        """
        if not isinstance(names, (list, tuple)):
            names = [names]
        ret = []
        for n in names:
            brats = PANCREAS_SEG(basedir, n)
            ret.extend(brats.load_3d())
        return ret