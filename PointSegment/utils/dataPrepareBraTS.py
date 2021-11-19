from sklearn.neighbors import KDTree
from os.path import join, exists, dirname, abspath
import numpy as np
import pandas as pd
import os, sys, glob, pickle
import nibabel as nib
from multiprocessing import Process
import concurrent.futures
from tqdm import tqdm
from scipy import ndimage

BASE_DIR = dirname(abspath(__file__))
ROOT_DIR = dirname(BASE_DIR)

sys.path.append(BASE_DIR)
sys.path.append(ROOT_DIR)
from helper_ply import write_ply
from helper_tool import DataProcessing as DP
import time
typeimg = ['t1ce','t1', 'flair', 't2', 'seg']


sub_grid_size = 0.01


dataset_path = "/home/ubuntu/Research/3D_Med_Seg/Volume_3D/BraTS_data/MICCAI_BraTS2020_TrainingData/training/HGG/"
attention_mask_path =  "/home/ubuntu/Research/3D_Med_Seg/Volume_3D/BraTS_data/submission/tannh10/TanND_probs_nii/BraTS2018_val/val18_probs95/BraTS2018_val/"
original_pc_folder = '/home/ubuntu/Research/3D_Med_Seg/Point-Unet/dataset/BraTS2020/original_ply'
sub_pc_folder =      '/home/ubuntu/Research/3D_Med_Seg/Point-Unet/dataset/BraTS2020/input0.01'
if not exists(original_pc_folder):
    os.makedirs(original_pc_folder) 
if not exists(sub_pc_folder):
    os.makedirs(sub_pc_folder) 

out_format = '.ply'
list_ID = os.listdir(dataset_path)
parallel = True
dataTraining = True


def load_volume(ID):

    def itensity_normalize_one_volume(volume):
        """
        normalize the itensity of an nd volume based on the mean and std of nonzeor region
        inputs:
            volume: the input nd volume
        outputs:
            out: the normalized nd volume
        """
        pixels = volume[volume > 0]
        mean = pixels.mean()
        std  = pixels.std()
        out = (volume - mean)/std
        # random normal too slow
        #out_random = np.random.normal(0, 1, size = volume.shape)
        out_random = np.zeros(volume.shape)
        out[volume == 0] = out_random[volume == 0]
        return out
    
    path_volume = os.path.join(dataset_path, ID, ID)
    output_volume = np.empty((5,240,240,155))

    #Load and process image
    for i,mod in enumerate(typeimg[:-1]):
        path_mod = str(path_volume+'_'+mod+'.nii.gz')
        img = np.asanyarray(nib.load(path_mod).dataobj)
        img = itensity_normalize_one_volume(img)
        output_volume[i] = img
    
    if dataTraining:
        path_mod = str(path_volume+'_'+typeimg[-1]+'.nii.gz')
        img = np.asanyarray(nib.load(path_mod).dataobj)
        img[img==4]=3
        output_volume[4] = img
    else:
        path_mask = os.path.join(attention_mask_path, ID+'.nii.gz')
        mask = np.asanyarray(nib.load(path_mask).dataobj)
        mask = mask.astype(np.uint8)
        output_volume[4] = mask
    
    return output_volume


def convert_pc2ply(volume,ID):
    
    channel,x_axis, y_axis, z_axis = volume.shape 
    data_list = [[x,y,z,volume[0][x][y][z],volume[1][x][y][z],volume[2][x][y][z],volume[3][x][y][z],volume[4][x][y][z]] for x in range(x_axis) for y in range(y_axis) for z in range(z_axis) if (volume[0][x][y][z] != 0 or volume[1][x][y][z] != 0 or volume[2][x][y][z] != 0 or volume[3][x][y][z] != 0)]


    pc_data = np.array(data_list)
    xyz_origin = pc_data[:,:3].astype(int)
    np.save(os.path.join(sub_pc_folder, ID+"_xyz_origin.npy"), xyz_origin)
    
    xyz_min = np.array([x_axis,y_axis,z_axis])


    pc_data[:, 0:3] /= xyz_min
    xyz = pc_data[:, :3].astype(np.float32)
    colors = pc_data[:, 3:7].astype(np.float32)
    labels = pc_data[:,7].astype(np.uint8)


    (unique, counts) = np.unique(labels, return_counts=True)
    print(ID," n point ", len(labels),unique, counts ) 


    #write full ply
    write_ply(os.path.join(original_pc_folder, ID+out_format), (xyz, colors, labels), ['x', 'y', 'z', 't1ce', 't1', 'flair', 't2' ,'class'])

    # save sub_cloud and KDTree file
    sub_xyz, sub_colors, sub_labels = DP.grid_sub_sampling(xyz, colors, labels, sub_grid_size)

    #write sub ply 
    write_ply(os.path.join(sub_pc_folder, ID+out_format), [sub_xyz, sub_colors, sub_labels], ['x', 'y', 'z', 't1ce', 't1', 'flair', 't2' ,'class'])

    kd_tree_file = os.path.join(sub_pc_folder, ID+ '_KDTree.pkl')
    search_tree = KDTree(sub_xyz)
    with open(kd_tree_file, 'wb') as f:
        pickle.dump(search_tree, f)

    proj_idx = np.squeeze(search_tree.query(xyz, return_distance=False))
    proj_idx = proj_idx.astype(np.int32)
    proj_save = os.path.join(sub_pc_folder, ID+ '_proj.pkl')
    with open(proj_save, 'wb') as f:
        pickle.dump([proj_idx, labels], f)


def process_data_and_save(ID):
    convert_pc2ply(load_volume(ID),ID)


if __name__ == '__main__':


    if parallel:
        with concurrent.futures.ProcessPoolExecutor(50) as executor:
            tqdm(executor.map(process_data_and_save, list_ID), total=len(list_ID))

    else:
        for i,ID in enumerate(list_ID):
            # conver_point_vs(load_volume(ID),ID)
            process_data_and_save(ID)
            # exit()
        
