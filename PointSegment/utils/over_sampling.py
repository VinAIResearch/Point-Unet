import os 
import nibabel as nib
import numpy as np
from scipy import ndimage

import sys
sys.path.append(os.path.split(sys.path[0])[0])
import shutil
from time import time
from tqdm import tqdm
import SimpleITK as sitk
import scipy.ndimage as ndimage

path_all_truth = "/home/ubuntu/Research/dataset/Pancreas-CT_processed_v1/seg/"
path_all_pred = "/home/ubuntu/Research/Brain_Point/RandLA-Net/Model_log/normalize_xyz/Pancreas/nii_Tannd/all_nii_fullsize/"
def over_binary(label0):
    new_label_level = label0.copy()
    N = np.count_nonzero(label0 ==1)
    print("l0 ", np.count_nonzero(label0 ==1))

    label1 = ndimage.binary_dilation(label0).astype(label0.dtype)
    N1 = np.count_nonzero(label1 ==1)
    label_level1 = label1 - label0
    print("l1 ", np.count_nonzero(label_level1 ==1))
    new_label_level[label_level1 == 1] = 2


    coord_X,coord_Y,coord_Z = np.where(label1 == 1)
    x_min,x_max = np.amin(coord_X),np.amax(coord_X)
    y_min,y_max = np.amin(coord_Y),np.amax(coord_Y)
    z_min,z_max = np.amin(coord_Z),np.amax(coord_Z)
    print("box ",(x_max-x_min)*(y_max-y_min)*(z_max-z_min))
    label2 = label1.copy()
    label2[x_min:x_max, y_min:y_max,z_min:z_max]=1

    label_level2 = label2 - label1
    print("l2 ", np.count_nonzero(label_level2 ==1))
    new_label_level[label_level2 == 1] = 3

    (unique, counts) = np.unique(new_label_level, return_counts=True)
    frequencies = np.asarray((unique, counts)).T
    print("queried_pc_labels ", frequencies)

    print(np.unique(new_label_level))
    return new_label_level,N1


# max_tumor = 0
# for ID in list_ID:
#     path_ID = os.path.join(path_seg,ID)
#     label0 = np.asanyarray(nib.load(path_ID).dataobj)
#     _, N_over = over_binary(label0)
#     if max_tumor < N_over:
#         max_tumor = N_over 
#         print("max_tumor ",max_tumor)
#     print("max_tumor ",max_tumor)

def dilation_over_truth(pred, truth):


    pred = ndimage.binary_dilation(pred).astype(pred.dtype)
    pred = np.logical_or(pred,truth) 


    return pred
    # print("miss ",)


    # print("pred0 ",np.count_nonzero(pred ==1))

    # pred = ndimage.binary_dilation(pred).astype(pred.dtype)
    # print("pred1 ",np.count_nonzero(pred ==1))

    # pred = ndimage.binary_dilation(pred).astype(pred.dtype)
    # print("pred2 ",np.count_nonzero(pred ==1))

    # pred = ndimage.binary_dilation(pred).astype(pred.dtype)
    # print("pred3 ",np.count_nonzero(pred ==1))

    # pred = ndimage.binary_dilation(pred).astype(pred.dtype)
    # print("pred4 ",np.count_nonzero(pred ==1))



list_ID =  os.listdir(path_all_truth)
n_point = 0
ID_max = None
# list_ID = ["PANCREAS_0058.nii.gz"]
for ID in list_ID:
    path_pred = os.path.join(path_all_pred,"PANCREAS_"+ID[5:9]+".nii.gz")
    path_truth = os.path.join(path_all_truth,ID)

    pred = np.asanyarray(nib.load(path_pred).dataobj)
    truth = np.asanyarray(nib.load(path_truth).dataobj)
    dilation_pred = dilation_over_truth(pred, truth)
    
    N = np.count_nonzero(dilation_pred ==1)

    if N > n_point:
        n_point = N
        print(ID,n_point)










# new_seg_dir = "/home/ubuntu/Research/dataset"

# ct = sitk.ReadImage("/home/ubuntu/Research/dataset/Pancreas-CT_processed_v1/ct/PANCREAS_0036.nii.gz", sitk.sitkInt16)
# label = sitk.ReadImage("/home/ubuntu/Research/dataset/Pancreas-CT_processed_v1/seg/label0036.nii.gz", sitk.sitkInt16)

# new_seg = sitk.GetImageFromArray(label)

# new_seg.SetDirection(ct.GetDirection())
# new_seg.SetOrigin(ct.GetOrigin())
# new_seg.SetSpacing(ct.GetSpacing())
# # new_seg.SetSpacing((ct.GetSpacing()[0] * int(1 / down_scale), ct.GetSpacing()[1] * int(1 / down_scale), slice_thickness / down_scale))


# sitk.WriteImage(new_seg, os.path.join(new_seg_dir,"label36.nii.gz" ))