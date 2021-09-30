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
# typeimg = ['t1ce','t1', 'flair', 't2']
struct1 = ndimage.generate_binary_structure(rank=3, connectivity=1)

# print("struct1 ",struct1)

task = "testing"
path_save_point = "/vinai/vuonghn/Research/Brain_Point/RandLA-Net/Model_log/obj/"

sub_grid_size = 0.01
# dataset_path = "/vinai/vuonghn/Research/BraTS/BraTS_data/BraTS2019_all/training/"
# dataset_path = "/vinai/vuonghn/Research/BraTS/BraTS_data/BraTS2019_all/val/"
# dataset_path_seg = "/vinai/vuonghn/Research/BraTS/BraTS_data/submission/submission3/val_weight_unet60_randlanet128/compare_all/"
# dataset_path = "/vinai/vuonghn/Research/BraTS/BraTS_data/MICCAI_BraTS2020_TestingData"

# list_probs = [0.35, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
list_probs = [0.35, 0.4, 0.5]
for probs in list_probs:
# probs = 95

    print("seting probs ", probs)
    dataset_path = "/vinai/vuonghn/Research/BraTS/BraTS_data/BraTS2018_all/val"
    original_pc_folder = '/vinai/vuonghn/Research/Brain_Point/RandLA-Net/Model_log/normalize_xyz/0.01_float/BraTS18_0.01_float/Point-Unet/input_66_val_attention/val2018_probs'+str(probs)+'_attention/original_ply/'
    sub_pc_folder =      '/vinai/vuonghn/Research/Brain_Point/RandLA-Net/Model_log/normalize_xyz/0.01_float/BraTS18_0.01_float/Point-Unet/input_66_val_attention/val2018_probs'+str(probs)+'_attention/input0.01/'

    Unet50k_mask =  "/vinai/vuonghn/Research/BraTS/BraTS_data/submission/tannh10/TanND_probs_nii/BraTS2018_val/val18_probs_"+str(probs)+"/"
    #Unet50k_mask =  "/vinai/vuonghn/Research/BraTS/3DUnet-Tensorflow-Brats18/train_log/train_log_round2/2019_full_random_160_208_176_class4/result_50k/result_validation/mask_probs_50k_on_val125_2019/mask_probs_50k_on_val125_2019_"+str(probs)+"/val2019_prob_unet_50k"
    # Unet50k_mask =  "/vinai/vuonghn/Research/BraTS/3DUnet-Tensorflow-Brats18/train_log/train_log_round2/2020_full_random_160_208_176_class4/result_50k/result_validation/mask_probs_50k_on_val125/probs_val_2020_50k_"+str(probs)+"/val_prob_unet_50k"
    # Unet50k_mask = '/vinai/vuonghn/Research/BraTS/3DUnet-Tensorflow-Brats18/train_log/train_log_round2/2020_full_random_160_208_176_class4/result_50k/result_validation/mask_probs_50k_on_val125/mask_probs_'+str(probs)+'_74/'
    # sub_pc_folder = '/home/ubuntu/BraTS20/normalize_xyz/full/training/input_'+str(sub_grid_size)
    if not exists(original_pc_folder):
        os.makedirs(original_pc_folder) 
    if not exists(sub_pc_folder):
        os.makedirs(sub_pc_folder) 


    if not exists(original_pc_folder):
        os.makedirs(original_pc_folder) 
    if not exists(sub_pc_folder):
        os.makedirs(sub_pc_folder) 

    out_format = '.ply'

    list_ID = os.listdir(dataset_path)


    # path_val_ID = "/vinai/vuonghn/Research/BraTS/3D_Medical_Segmentation/utils/data_splited/val_IDs_fold1_82_182.txt"
    # with open(path_val_ID) as f:
    #     content = f.readlines()
    # list_ID = [x.strip() for x in content] 


    # # print("list_ID ",list_ID, len(list_ID))
    # # exit()
    # total_ID = len(list_ID)
    # print("len ", total_ID)

    # exist_file = os.listdir(original_pc_folder)
    # print("exist_file ", len(exist_file))


    # for files in exist_file:
    #     files = files.split(".ply")[0]

    #     if files in list_ID:
    #         list_ID.remove(files)

    # list_ID = list_ID[9:18]

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
        output_volume = np.empty((6,240,240,155))

        #Load and process image
        for i,mod in enumerate(typeimg[:-1]):
            path_mod = str(path_volume+'_'+mod+'.nii.gz')
            img = np.asanyarray(nib.load(path_mod).dataobj)
            img = itensity_normalize_one_volume(img)



            output_volume[i] = img


        # # Load and process image
        # path_mod = str(path_volume+'_'+typeimg[-1]+'.nii.gz')
        # img = np.asanyarray(nib.load(path_mod).dataobj)
        # img[img==4]=3
        # output_volume[4] = img
        

        path_mask = os.path.join(Unet50k_mask, ID+'.nii.gz')
        mask = np.asanyarray(nib.load(path_mask).dataobj)
        mask[mask>0]=1
        mask = mask.astype(np.uint8)

        # print("unique mask ", np.unique(mask))
        # exit()
        # mask = np.empty((240,240,155))
        # mask[img>0]=1
        # print("mask ", mask.shape, np.unique(mask))
        # exit()

        # path_mod = os.path.join(dataset_path_seg,ID+'.nii.gz')
        # img = np.asanyarray(nib.load(path_mod).dataobj)
        # img[img>0]=1
        # img = ndimage.binary_dilation(img, struct1, iterations=2)

        
        output_volume[5] = mask
        
        return output_volume

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
        
        channel,x_axis, y_axis, z_axis = volume.shape #(5, 240, 240, 155)
        # print("xyz_min ", volume.shape)
        # exit()
        data_list = [[x,y,z,volume[0][x][y][z],volume[1][x][y][z],volume[2][x][y][z],volume[3][x][y][z],volume[4][x][y][z],volume[5][x][y][z]] for x in range(x_axis) for y in range(y_axis) for z in range(z_axis) if (volume[0][x][y][z] != 0 or volume[1][x][y][z] != 0 or volume[2][x][y][z] != 0 or volume[3][x][y][z] != 0)]


        pc_label = np.array(data_list)

        # pc_label = ["X","Y","Z","V1","V2","V3","V4","M"]


        xyz_origin = pc_label[:,:3].astype(int)
        np.save(os.path.join(sub_pc_folder, ID+"_xyz_origin.npy"), xyz_origin)
        


        # xyz_min = np.amin(pc_label, axis=0)[0:3]



        xyz_min = np.array([x_axis,y_axis,z_axis])


        pc_label[:, 0:3] /= xyz_min


        xyz = pc_label[:, :3].astype(np.float32)
        colors = pc_label[:, 3:7].astype(np.float32)
        labels = pc_label[:,7].astype(np.uint8)

        masks = pc_label[:,8].astype(np.uint8)
        print("labels ", np.unique(labels))
        print("colors ", np.unique(colors) , len(np.unique(colors)))

        #write full ply
        write_ply(os.path.join(original_pc_folder, ID+out_format), (xyz, colors, labels,masks), ['x', 'y', 'z', 't1ce', 't1', 'flair', 't2' ,'class','masks'])

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
        # print("process_data_and_save ", ID)
        convert_pc2ply(load_volume(ID),ID)


    # if __name__ == '__main__':

        # list_ID = ['BraTS20_Training_261']
        # Run with one core


    parallel = False
    if parallel:
        with concurrent.futures.ProcessPoolExecutor(20) as executor:
            tqdm(executor.map(process_data_and_save, list_ID), total=len(list_ID))

    else:
        for i,ID in enumerate(list_ID):
            # conver_point_vs(load_volume(ID),ID)
            process_data_and_save(ID)
            # exit()
        
        # # Run with milti core
        # procs = []
        # proc = Process(target=process_data_and_save)  # instantiating without any argument
        # procs.append(proc)
        # proc.start()

        # for name in list_ID:
        #     # print(name)
        #     proc = Process(target=process_data_and_save, args=(name,))
        #     procs.append(proc)
        #     proc.start()

        # # complete the processes
        # for proc in procs:
        #     proc.join()