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
typeimg = ['t1ce','t1', 'flair', 't2', 'seg']
# typeimg = ['t1ce','t1', 'flair', 't2']
struct1 = ndimage.generate_binary_structure(rank=3, connectivity=1)

# print("struct1 ",struct1)

task = "testing"

sub_grid_size = 0.01


dilation_123 = False
dilation_attention = True

dataset_path = "/vinai/vuonghn/Research/3D_Med_Seg/Point_3D/RandLA-Net/Model_log/normalize_xyz/Pancreas-CT_processed/Pancreas-CT_processed_v1"
data_nii = os.path.join(dataset_path,"ct")
label_nii = os.path.join(dataset_path,"seg")
original_pc_folder = '/vinai/vuonghn/Research/3D_Med_Seg/Point_3D/RandLA-Net/Model_log/normalize_xyz/Pancreas-CT_processed/Pancreas-CT_processed_v1/PointCloud/original_ply/'
sub_pc_folder =      '/vinai/vuonghn/Research/3D_Med_Seg/Point_3D/RandLA-Net/Model_log/normalize_xyz/Pancreas-CT_processed/Pancreas-CT_processed_v1/PointCloud/input0.01/'

# label_nii_attention  =   "/vinai/vuonghn/Research/Brain_Point/RandLA-Net/Model_log/normalize_xyz/Pancreas/nii_Tannd/all_nii_fullsize/"
# original_pc_folder = '/vinai/vuonghn/Research/Brain_Point/RandLA-Net/Model_log/normalize_xyz/Pancreas_v1/nii_Tannd/full_size/fold_1/fold1_probs0.5'
# sub_pc_folder =      '/vinai/vuonghn/Research/Brain_Point/RandLA-Net/Model_log/normalize_xyz/Pancreas_v1/nii_Tannd/full_size/fold_1/fold1_probs0.5'


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
    # path_label = os.path.join(label_nii,"PANCREAS_"+ID+".nii.gz")
    path_label = os.path.join(label_nii,"label"+ID+".nii.gz")
    img = np.asanyarray(nib.load(path_data).dataobj)
    # print("img 0",img.shape, np.amin(img), np.amax(img))
    img = itensity_normalize_one_volume(img)
    # print("img 1",img.shape, np.amin(img), np.amax(img))
    label = np.asanyarray(nib.load(path_label).dataobj)

    # if max_tumor < np.count_nonzero(label ==1): 
    #     max_tumor = np.count_nonzero(label ==1)


    # print("label ", label.shape)
    # exit()
    # print("label 0 ",label[229, 10, 133])
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
                            
                    # elif ground_truth[x][y][z] == 1:
                    #     point = "v "+str(x)+" "+str(y)+" "+str(z) + " 255 0 0 "+"\n"
                    # elif ground_truth[x][y][z] == 2:
                    #     point = "v "+str(x)+" "+str(y)+" "+str(z) + " 0 255 0 "+"\n"
                    # else:
                    #     point = "v "+str(x)+" "+str(y)+" "+str(z) + " 0 0 255"+ "\n"
                    else:
                        point = "v "+str(x)+" "+str(y)+" "+str(z) + " "+str(volume[0][x][y][z])+" "+str(volume[0][x][y][z])+ " "+str(volume[0][x][y][z])+"\n"
                        
                    
                    if x%2==0 and y%2==0 and z%2==0:
                        f.write(point)
    f.close()

def convert_pc2ply(volume,ID):

    img = volume[0]
    label = volume[1]
    print("label 1 ",label[229, 10, 133])
    
    x_axis, y_axis, z_axis = img.shape #(5, 240, 240, 155)
    # x_axis, y_axis, z_axis = 20,20,20
    # print("xyz_min ", volume.shape)
    # exit()
    # print("loop")
    data_list = [[x,y,z,img[x][y][z],label[x][y][z]] for x in range(x_axis) for y in range(y_axis) for z in range(z_axis) if (img[x][y][z] != 0)]
    # print("end-loop")

    pc_label = np.array(data_list)

    # pc_label = ["X","Y","Z","value","label"]


    xyz_origin = pc_label[:,:3].astype(int)
    np.save(os.path.join(sub_pc_folder, ID+"_xyz_origin.npy"), xyz_origin)
    


    # xyz_min = np.amin(pc_label, axis=0)[0:3]



    xyz_min = np.array([x_axis,y_axis,z_axis])


    pc_label[:, 0:3] /= xyz_min


    xyz = pc_label[:, :3].astype(np.float32)
    colors = pc_label[:, 3:4].astype(np.float32)
    labels = pc_label[:,4].astype(np.uint8)

    print("labels ", np.unique(labels),labels.shape)
    print("colors ", np.unique(colors) , len(np.unique(colors)),colors.shape)


    #write full ply
    write_ply(os.path.join(original_pc_folder, ID+out_format), (xyz, colors, labels), ['x', 'y', 'z','value' ,'class'])

    # save sub_cloud and KDTree file


    (unique, counts) = np.unique(labels, return_counts=True)


    sub_xyz, sub_colors, sub_labels = DP.grid_sub_sampling(xyz, colors, labels, sub_grid_size)
    # sub_xyz, sub_colors, sub_labels = xyz, colors, labels
    print("sub_labels ", np.unique(sub_labels))
    print("sub_colors ", len(np.unique(sub_colors)))



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

    n_point = 180000
    # n_point = 30000
    loop = 8

    img = volume[0]
    label = volume[1]

    print("img ", img.shape)
    
    x_axis, y_axis, z_axis = img.shape #(5, 240, 240, 155)
    print("loop ",ID)
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
    # print("label 3 ",label[229, 10, 133])
    # print("labels",labels[(114176*229)+ (223*10) + (133)])
    # print("xyz",xyz[(114176*229)+ (223*10) + (133)])



    for i in range(loop):

        queried_idx = tumor + random.sample(none_tumor, k=n_point - len(tumor))
        queried_idx = np.array(queried_idx)
        sampling_xyz = xyz[queried_idx].astype(np.uint16)
        sampling_colors = colors[queried_idx]
        sampling_labels = labels[queried_idx]
        np.save(os.path.join(sub_pc_folder, ID+"_xyz_origin_loop_"+str(i)+".npy"), sampling_xyz)
        sampling_xyz = sampling_xyz.astype(np.float32)/xyz_min
        
        # print("sampling_xyz 1",sampling_xyz)
        # print("labels ", np.unique(sampling_labels),sampling_labels.shape)
        # print("colors ", np.unique(sampling_colors) , len(np.unique(sampling_colors)),sampling_colors.shape)

        # print("N: ", len(queried_idx))
        # #write full ply
        name_loop = str(ID)+"_loop_"+str(i)
        write_ply(os.path.join(original_pc_folder, name_loop+out_format), (sampling_xyz, sampling_colors, sampling_labels), ['x', 'y', 'z','value' ,'class'])

    print("Done ",ID)

def over_binary(label0):
    new_label_level = label0.copy()
    # N = np.count_nonzero(label0 ==1)
    # print("l0 ", np.count_nonzero(label0 ==1))

    label1 = ndimage.binary_dilation(label0).astype(label0.dtype)
    # N1 = np.count_nonzero(label1 ==1)
    label_level1 = label1 - label0
    # print("l1 ", np.count_nonzero(label_level1 ==1))
    new_label_level[label_level1 == 1] = 2


    coord_X,coord_Y,coord_Z = np.where(label1 == 1)
    x_min,x_max = np.amin(coord_X),np.amax(coord_X)
    y_min,y_max = np.amin(coord_Y),np.amax(coord_Y)
    z_min,z_max = np.amin(coord_Z),np.amax(coord_Z)
    # print("box ",(x_max-x_min)*(y_max-y_min)*(z_max-z_min))
    label2 = label1.copy()
    label2[x_min:x_max, y_min:y_max,z_min:z_max]=1

    label_level2 = label2 - label1
    # print("l2 ", np.count_nonzero(label_level2 ==1))
    new_label_level[label_level2 == 1] = 3

    # (unique, counts) = np.unique(new_label_level, return_counts=True)
    # frequencies = np.asarray((unique, counts)).T
    # print("queried_pc_labels ", frequencies)

    # print(np.unique(new_label_level))
    return new_label_level

def sampling_over_convert_pc2ply(volume,ID):

    n_point = 180000
    # n_point = 30000
    loop = 8

    img = volume[0]
    label = volume[1]
    mask_over = over_binary(label)

    (unique, counts) = np.unique(mask_over, return_counts=True)
    frequencies = np.asarray((unique, counts)).T

    

    x_axis, y_axis, z_axis = img.shape #(5, 240, 240, 155)

    data_list = [[x,y,z,img[x][y][z],label[x][y][z],mask_over[x][y][z]] for x in range(x_axis) for y in range(y_axis) for z in range(z_axis)]



    pc_label = np.array(data_list)
    xyz_min = np.array([x_axis,y_axis,z_axis])
    xyz_min = xyz_min.astype(np.float32)
    # pc_label[:, 0:3] /= xyz_min
    xyz = pc_label[:, :3].astype(np.uint16)
    colors = pc_label[:, 3:4].astype(np.float32)
    labels = pc_label[:,4].astype(np.uint8)
    mask = pc_label[:,5].astype(np.uint8)
    



    
    
    
    none_tumor = list(np.where(mask == 0)[0])
    tumor_1 = list(np.where(mask ==1)[0])
    tumor_2 = list(np.where(mask ==2)[0])
    tumor_3 = list(np.where(mask ==3)[0])
    # print("label 3 ",label[229, 10, 133])
    # print("labels",labels[(114176*229)+ (223*10) + (133)])
    # print("xyz",xyz[(114176*229)+ (223*10) + (133)])



    for i in range(loop):
        k_3 = int((n_point - len(tumor_1) - len(tumor_2))*0.8)
        k_0 = n_point - k_3 - len(tumor_1) - len(tumor_2)

        # print(len(tumor_1), len(tumor_2), k_3, k_0)
        
        queried_idx = tumor_1 + tumor_2 + random.sample(tumor_3, k=k_3) + random.sample(none_tumor, k=k_0)
        queried_idx = np.array(queried_idx)

        sampling_xyz = xyz[queried_idx].astype(np.uint16)
        sampling_colors = colors[queried_idx]
        sampling_labels = labels[queried_idx]
        np.save(os.path.join(sub_pc_folder, ID+"_xyz_origin_loop_"+str(i)+".npy"), sampling_xyz)
        sampling_xyz = sampling_xyz.astype(np.float32)/xyz_min
        
        # print("sampling_xyz 1",sampling_xyz)
        # print("labels ", np.unique(sampling_labels),sampling_labels.shape)
        # print("colors ", np.unique(sampling_colors) , len(np.unique(sampling_colors)),sampling_colors.shape)


        # #write full ply
        name_loop = str(ID)+"_loop_"+str(i)
        write_ply(os.path.join(original_pc_folder, name_loop+out_format), (sampling_xyz, sampling_colors, sampling_labels), ['x', 'y', 'z','value' ,'class'])

    print("Done ",ID)

def dilation_over_truth(pred, truth):
    pred = ndimage.binary_dilation(pred).astype(pred.dtype)
    pred = np.logical_or(pred,truth) 
    return pred



def process_data_and_save(ID):
    sampling_convert_pc2ply(load_volume(ID),ID)
process_data_and_save("0001")
exit()

parallel = True
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
        process_data_and_save(ID)
        # exit()


print("DONE")