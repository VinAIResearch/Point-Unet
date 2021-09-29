import nibabel as nib
import numpy as np
import os
import matplotlib.pyplot as plt
import random
from sklearn.model_selection import train_test_split
from statistics import mean 
from shutil import copyfile
from visual_truth_pred import showImage,showImage5
import csv


def dice_coefficient(truth, prediction):
    # print("truth : ",np.sum(truth) , "prediction : ",np.sum(prediction))
    if (np.sum(truth) + np.sum(prediction)) == 0:
        return 1
    return 2 * np.sum(truth * prediction)/(np.sum(truth) + np.sum(prediction))


def preprocess_label(label):
    bachground = label == 0
    ncr = label == 1  # Necrotic and Non-Enhancing Tumor (NCR/NET)
    ed = label == 2  # Peritumoral Edema (ED)
    et = label == 3  # GD-enhancing Tumor (ET)
    return np.array([bachground, ncr, ed, et], dtype=np.uint8)

def visual_metric(path_imgs, path_preds_1,path_preds_2, path_save_compare, path_truths=None):
    all_dice = {'ID':[],'bachground':[],'ncr':[], 'ed':[] , 'et':[]}
    name_labels = ['bachground','ncr', 'ed' , 'et']
    path_report = os.path.join(path_save_compare,"Dise_score.csv" ) 
    
    list_IDs = os.listdir(path_preds_1)
    with open(path_report, 'w') as csvfile:

        #fieldnames = ['url', 'ground_truth','predict_result','time_predict','number_of_frames']
        fieldnames = ['ID', 'ET_1','WT_2','TC_4']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()


                    


        for i_image, ID in enumerate(list_IDs):

            ID = ID.split(".nii.gz")[0]

            all_dice["ID"].append(ID)
            if path_truths is None:  
                truth_seg = None
            else:
                path_truth = os.path.join(path_truths,ID,ID+"_seg.nii.gz") 
                truth_seg = np.asanyarray(nib.load(path_truth).dataobj)
                truth_seg[truth_seg==4]=3
            path_pred_1 = os.path.join(path_preds_1,ID+".nii.gz")
            path_pred_2 = os.path.join(path_preds_2,ID+".nii.gz")



            pred_seg_1 = np.asanyarray(nib.load(path_pred_1).dataobj)
            pred_seg_1[pred_seg_1==4]=3
            pred_seg_2 = np.asanyarray(nib.load(path_pred_2).dataobj)
            pred_seg_2[pred_seg_2==4]=3

            path_img = os.path.join(path_imgs,ID,ID+"_flair.nii.gz")
            img3d = np.asanyarray(nib.load(path_img).dataobj)

            if truth_seg is not None:
                ani = showImage5(img3d,0,truth_seg,pred_seg_1,pred_seg_2)
                ani.save(os.path.join(path_save_compare,ID+'.gif'), writer='imagemagick', fps=5)

                truth_seg = preprocess_label(truth_seg)
                pred_seg_1 = preprocess_label(pred_seg_1)
                pred_seg_2 = preprocess_label(pred_seg_2)

                for i, name_label in enumerate(name_labels):
                    dice = dice_coefficient(truth_seg[i],pred_seg_1[i])
                    all_dice[name_label].append(round(dice,5))
        
                print(i_image, " / ", len(list_IDs), 'ID:   ', all_dice["ID"][i_image],' dice_ncr:   ',all_dice["ncr"][i_image],'dice_ed:   ',all_dice["ed"][i_image],'dice_et:   ',all_dice["et"][i_image])
                writer.writerow({'ID': ID, 'ET_1': all_dice["ncr"][i_image], 'WT_2': all_dice["ed"][i_image], 'TC_4': all_dice["et"][i_image]})
            if truth_seg is None:
                print("writing ",i_image, " / ", len(list_IDs))
                ani = showImage5(img3d,0,pred_seg_1,pred_seg_2)
                ani.save(os.path.join(path_save_compare,ID+'.gif'), writer='imagemagick', fps=5)





            



        # while 1:
        print("mean bachground - ncr -  ed - et -: ", mean(all_dice["bachground"]) , mean(all_dice["ncr"]),mean(all_dice["ed"]),mean(all_dice["et"]))


# path_imgs = "/vinai/vuonghn/Research/BraTS/BraTS_data/MICCAI_BraTS2020_TrainingData/training/HGG/"
# path_truths = "/vinai/vuonghn/Research/BraTS/BraTS_data/MICCAI_BraTS2020_TrainingData/training/HGG/"
# path_seg = "/vinai/vuonghn/Research/BraTS/3DUnet-Tensorflow-Brats18/save_pred/seg_nii/train20_set6_lr_0.01_model_60000"
# path_save_compare = "/vinai/vuonghn/Research/BraTS/3DUnet-Tensorflow-Brats18/save_pred/compare/train20_set6_lr_0.01_model_60000"

# path_imgs = "/vinai/vuonghn/Research/BraTS/BraTS_data/MICCAI_BraTS2020_ValidationData/val"
# path_truths = None
# path_seg = "/vinai/vuonghn/Research/BraTS/3DUnet-Tensorflow-Brats18/save_pred/seg_nii/val20_set6_lr_0.01_model_60000_modify_ET4=0"
# path_save_compare = "/vinai/vuonghn/Research/BraTS/3DUnet-Tensorflow-Brats18/save_pred/compare/val20_set6_lr_0.01_model_60000_modify_ET4=0"

path_imgs = "/vinai/vuonghn/Research/BraTS/BraTS_data/MICCAI_BraTS2020_TrainingData/training/HGG/"
path_truths = "/vinai/vuonghn/Research/BraTS/BraTS_data/MICCAI_BraTS2020_TrainingData/training/HGG/"
path_seg = "/vinai/vuonghn/Brain_Point/RandLA-Net/Model_log/normalize_xyz/0.01/output/train_test_49600_0.01_1_1_1_1_bs16"
path_seg2 = "/vinai/vuonghn/Research/BraTS/3DUnet-Tensorflow-Brats18/save_pred/seg_nii/train20_set6_lr_0.01_model_60000"
# path_save_compare = "/vinai/vuonghn/Brain_Point/RandLA-Net/Model_log/normalize_xyz/0.01/output/visual_train_test_49600_0.01_1_1_1_1_bs16"
path_save_compare = "/vinai/vuonghn/compare_result/"


if  not os.path.exists(path_save_compare):
    os.mkdir(path_save_compare) 
visual_metric(path_imgs,path_seg,path_seg2,path_save_compare, path_truths)