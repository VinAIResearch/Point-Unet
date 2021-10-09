import nibabel as nib
import numpy as np
import os
import matplotlib.pyplot as plt
import random
from sklearn.model_selection import train_test_split
from statistics import mean 
from shutil import copyfile
import csv
import concurrent.futures
from tqdm import tqdm
import SimpleITK as itk 
from scipy import ndimage
from medpy import metric
from scipy.spatial.distance import directed_hausdorff
import numpy
from scipy.ndimage import _ni_support
from scipy.ndimage.morphology import distance_transform_edt, binary_erosion,\
    generate_binary_structure
from scipy.ndimage.measurements import label, find_objects
from scipy.stats import pearsonr

def dice_coefficient(truth, prediction):
    if (np.sum(truth) + np.sum(prediction)) == 0:
        return 1
    return 2 * np.sum(truth * prediction)/(np.sum(truth) + np.sum(prediction))

def preprocess_label(label):
    bachground = label == 0
    ncr = label == 1  # Necrotic and Non-Enhancing Tumor (NCR/NET)
    ed = label == 2  # Peritumoral Edema (ED)
    et = label == 4  # GD-enhancing Tumor (ET)
    WT = ncr+ed+et
    TC = ncr + et
    ET = et
    return np.array([bachground, WT, TC, ET], dtype=np.uint8)

def calculate_dice(path_truth, path_pred, path_report):
    all_dice = {'ID':[],'bachground':[],'ncr':[], 'ed':[] , 'et':[]}
    name_labels = ['bachground','ncr', 'ed' , 'et']
    list_IDs = os.listdir(path_pred)
    with open(path_report, 'w') as csvfile:
        fieldnames = ['ID', 'ET_1','WT_2','TC_4']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        for i_image, ID in enumerate(list_IDs):
            ID = ID.split(".nii.gz")[0]
            all_dice["ID"].append(ID)
            in_path_truth = os.path.join(path_truth,ID,ID+"_seg.nii.gz") 
            truth_seg = np.asanyarray(nib.load(in_path_truth).dataobj)
            # # Load by nii
            in_path_pred = os.path.join(path_pred,ID+".nii.gz")
            pred_seg = np.asanyarray(nib.load(in_path_pred).dataobj)

            truth_seg = preprocess_label(truth_seg)
            pred_seg = preprocess_label(pred_seg)

            for i, name_label in enumerate(name_labels):
                dice = dice_coefficient(truth_seg[i],pred_seg[i])
                all_dice[name_label].append(round(dice,5))

            print(i_image, " / ", len(list_IDs), 'ID:   ', all_dice["ID"][i_image],' WT:   ',all_dice["ncr"][i_image],'TC:   ',all_dice["ed"][i_image],'ET:   ',all_dice["et"][i_image])
            writer.writerow({'ID': ID, 'ET_1': all_dice["ncr"][i_image], 'WT_2': all_dice["ed"][i_image], 'TC_4': all_dice["et"][i_image]})
        print("Dice WT -  TC - ET : ", mean(all_dice["ncr"]),mean(all_dice["ed"]),mean(all_dice["et"]))



path_truths ="/vinai/vuonghn/Research/3D_Med_Seg/Volume_3D/BraTS_data/MICCAI_BraTS2020_TrainingData/training/HGG"
path_seg = '/vinai/vuonghn/Research/3D_Med_Seg/Volume_3D/BraTS_data/submission/submission4/RandLANet_74case_input64k_train295'
path_report = os.path.join("/vinai/vuonghn/Research/3D_Med_Seg/Point-Unet/dataset/BraTS2020","Offline"+".csv" ) 

calculate_dice(path_truths,path_seg, path_report)
exit()

