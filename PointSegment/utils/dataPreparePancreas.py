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
import random




sub_grid_size = 0.01

dataset_path = "/home/ubuntu/Research/3D_Med_Seg/Point_3D/RandLA-Net/Model_log/normalize_xyz/Pancreas-CT_processed/Pancreas-CT_processed_v1"
data_nii = os.path.join(dataset_path,"ct")
label_nii = os.path.join(dataset_path,"seg")
original_pc_folder = '/home/ubuntu/Research/3D_Med_Seg/Point_3D/RandLA-Net/Model_log/normalize_xyz/Pancreas-CT_processed/Pancreas-CT_processed_v1/PointCloud/original_ply/'
sub_pc_folder =      '/home/ubuntu/Research/3D_Med_Seg/Point_3D/RandLA-Net/Model_log/normalize_xyz/Pancreas-CT_processed/Pancreas-CT_processed_v1/PointCloud/input0.01/'



if not exists(original_pc_folder):
    os.makedirs(original_pc_folder) 
if not exists(sub_pc_folder):
    os.makedirs(sub_pc_folder) 

out_format = '.ply'
list_ID = os.listdir(data_nii)

max_tumor = 0
def load_volume(ID):
    global max_tumor

    def itensity_normalize_one_volume(volume):
        """
        normalize the itensity of an nd volume based on the mean and std of nonzeor region
        inputs:
            volume: the input nd volume
        outputs:
            out: the normalized nd volume
        """
        pixels = volume
        mean = pixels.mean()
        std  = pixels.std()
        out = (volume - mean)/std
        return out
    
    path_volume = os.path.join(dataset_path, ID, ID)



    path_data = os.path.join(data_nii,"PANCREAS_"+ID+".nii.gz")
    path_label = os.path.join(label_nii,"label"+ID+".nii.gz")
    img = np.asanyarray(nib.load(path_data).dataobj)
    img = itensity_normalize_one_volume(img)
    label = np.asanyarray(nib.load(path_label).dataobj)


    return [img,label]

def conver_point_vs(volume,ID):
    volume = volume.astype(np.uint8)
    ground_truth = volume[4].copy()
    print("unique label ", np.unique(volume[4]))

    f = open(path_save_point+ID+".obj", "w")
    channel,x_axis, y_axis, z_axis = volume.shape
    # volume = (volume*255)/np.amax(volume)
    for x in range(x_axis):
        for y in range(y_axis):
            for z in range(z_axis):
                if (volume[0][x][y][z] != 0 or volume[1][x][y][z] != 0 or volume[2][x][y][z] != 0 or volume[3][x][y][z] != 0):
                    if ground_truth[x][y][z] == 0:
                        # point = "v "+str(x)+" "+str(y)+" "+str(z) + " 0 0 0 "+"\n"
                        if (x%3==0 and y%4==0 and z%5==0) or (z%3==0 and x%4==0 and y%5==0):
                            point = "v "+str(x)+" "+str(y)+" "+str(z) + " "+str(volume[0][x][y][z])+" "+str(volume[0][x][y][z])+ " "+str(volume[0][x][y][z])+"\n"
                        else:
                            continue
                    else:
                        point = "v "+str(x)+" "+str(y)+" "+str(z) + " "+str(volume[0][x][y][z])+" "+str(volume[0][x][y][z])+ " "+str(volume[0][x][y][z])+"\n"
                        
                    
                    if x%2==0 and y%2==0 and z%2==0:
                        f.write(point)
    f.close()

def convert_pc2ply(volume,ID):

    img = volume[0]
    label = volume[1]
    x_axis, y_axis, z_axis = img.shape #(5, 240, 240, 155)
    data_list = [[x,y,z,img[x][y][z],label[x][y][z]] for x in range(x_axis) for y in range(y_axis) for z in range(z_axis) if (img[x][y][z] != 0)]
    pc_label = np.array(data_list)
    xyz_origin = pc_label[:,:3].astype(int)
    np.save(os.path.join(sub_pc_folder, ID+"_xyz_origin.npy"), xyz_origin)
    xyz_min = np.array([x_axis,y_axis,z_axis])
    pc_label[:, 0:3] /= xyz_min

    xyz = pc_label[:, :3].astype(np.float32)
    colors = pc_label[:, 3:4].astype(np.float32)
    labels = pc_label[:,4].astype(np.uint8)


    write_ply(os.path.join(original_pc_folder, ID+out_format), (xyz, colors, labels), ['x', 'y', 'z','value' ,'class'])

    (unique, counts) = np.unique(labels, return_counts=True)

    sub_xyz, sub_colors, sub_labels = DP.grid_sub_sampling(xyz, colors, labels, sub_grid_size)
    (unique_sub, counts_sub) = np.unique(sub_labels, return_counts=True)


    print(ID," len(    xyz) ", len(labels),unique, counts ) 
    print(ID," len(sub_xyz) ", len(sub_labels),unique_sub, counts_sub)


    
    #write sub ply 
    write_ply(os.path.join(sub_pc_folder, ID+out_format), [sub_xyz, sub_colors, sub_labels], ['x', 'y', 'z' ,'value','class'])

    kd_tree_file = os.path.join(sub_pc_folder, ID+ '_KDTree.pkl')
    search_tree = KDTree(sub_xyz)
    with open(kd_tree_file, 'wb') as f:
        pickle.dump(search_tree, f)

    proj_idx = np.squeeze(search_tree.query(xyz, return_distance=False))
    proj_idx = proj_idx.astype(np.int32)
    proj_save = os.path.join(sub_pc_folder, ID+ '_proj.pkl')
    with open(proj_save, 'wb') as f:
        pickle.dump([proj_idx, labels], f)


def sampling_convert_pc2ply(volume,ID):

    # n_point = 180000
    n_point = 230000
    loop = 8

    img = volume[0]
    label = volume[1]

    
    x_axis, y_axis, z_axis = img.shape #(5, 240, 240, 155)
    print("loop ",ID, img.shape)
    data_list = [[x,y,z,img[x][y][z],label[x][y][z]] for x in range(x_axis) for y in range(y_axis) for z in range(z_axis)]
    
    pc_label = np.array(data_list)
    xyz_min = np.array([x_axis,y_axis,z_axis])
    xyz_min = xyz_min.astype(np.float32)
    # pc_label[:, 0:3] /= xyz_min
    xyz = pc_label[:, :3].astype(np.uint16)
    colors = pc_label[:, 3:4].astype(np.float32)
    labels = pc_label[:,4].astype(np.uint8)
    
    none_tumor = list(np.where(labels == 0)[0])
    tumor = list(np.where(labels > 0)[0])

    for i in range(loop):

        queried_idx = tumor + random.sample(none_tumor, k=n_point - len(tumor))
        queried_idx = np.array(queried_idx)
        sampling_xyz = xyz[queried_idx].astype(np.uint16)
        sampling_colors = colors[queried_idx]
        sampling_labels = labels[queried_idx]
        np.save(os.path.join(sub_pc_folder, ID+"_xyz_origin_loop_"+str(i)+".npy"), sampling_xyz)
        sampling_xyz = sampling_xyz.astype(np.float32)/xyz_min
        
        name_loop = str(ID)+"_loop_"+str(i)
        write_ply(os.path.join(original_pc_folder, name_loop+out_format), (sampling_xyz, sampling_colors, sampling_labels), ['x', 'y', 'z','value' ,'class'])
        print("save  ",os.path.join(original_pc_folder, name_loop+out_format))


def process_data_and_save(ID):
    sampling_convert_pc2ply(load_volume(ID),ID)

parallel = False
if parallel:
    ID_s = []
    for i in range(1, len(list_ID)+1): 
        ID = str(i).zfill(4)
        ID_s.append(ID)
    with concurrent.futures.ProcessPoolExecutor(20) as executor:
        tqdm(executor.map(process_data_and_save, ID_s), total=len(ID_s))

else:
    for i in range(1, len(list_ID),4):
        ID = str(i).zfill(4)
        if i ==0:
            continue
        print("ID ", ID)
        process_data_and_save(ID)
       


print("DONE")