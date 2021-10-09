import numpy as np
import os
import SimpleITK as sitk
import nibabel as nib
import math
import csv
from shutil import copyfile


LABEL_PATH = '/vinai/vuonghn/Research/3D_Med_Seg/Point_3D/RandLA-Net/Model_log/normalize_xyz/Pancreas-CT_processed/Pancreas-CT_processed_v1/seg'
PREDICTION_PATH = '/home/ubuntu/vuonghn/predictPancreas'
path_save_csv = "/vinai/vuonghn/Research/3D_Med_Seg/Point-Unet/dataset/Pancreas/diceloss.csv"

def preprocess_label(label):
    bachground = label == 0
    pancreas = label == 1  
    return np.array([bachground, pancreas], dtype=np.uint8)

def dice_coefficient(truth, prediction):
    if (np.sum(truth) + np.sum(prediction)) == 0:
        return 1
    dice = 2 * np.sum(truth * prediction)/(np.sum(truth) + np.sum(prediction) + 1e-10)
    if math.isnan(dice):
        dice = 0
    return dice

def binary_dice3d(s,g):
    """
    dice score of 3d binary volumes
    inputs: 
        s: segmentation volume
        g: ground truth volume
    outputs:
        dice: the dice score
    """
    assert(len(s.shape)==3)
    [Ds, Hs, Ws] = s.shape
    [Dg, Hg, Wg] = g.shape
    assert(Ds==Dg and Hs==Hg and Ws==Wg)
    prod = np.multiply(s, g)
    s0 = prod.sum()
    s1 = s.sum()
    s2 = g.sum()
    dice = (2.0*s0 + 1e-10)/(s1 + s2 + 1e-10)
    return dice


with open(path_save_csv, 'w') as csvfile:
    fieldnames = ['file_name', 'Dice']
    writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
    writer.writeheader()

    dice_all_data = []
    index = 1
    for filename in os.listdir(LABEL_PATH):
        ID = filename[5:9]
        max_dice_ID = 0
        path_pred_max = None
    
        path_truth = os.path.join(LABEL_PATH, filename)
        path_pred  = os.path.join(PREDICTION_PATH, ID+"_loop_"+str(0)+".nii.gz")
        truth = np.asanyarray(nib.load(path_truth).dataobj)
        pred_vuong = np.asanyarray(nib.load(path_pred).dataobj)


        truth_seg = preprocess_label(truth)
        pred_seg = preprocess_label(pred_vuong)
        
        dice = dice_coefficient(truth_seg[1],pred_seg[1])
        print(ID , "     ", dice)
        writer.writerow({'file_name': filename, 'Dice': round(dice,5)})
        dice_all_data.append(dice)
        index +=1
    print('dice mean', np.mean(dice_all_data))

