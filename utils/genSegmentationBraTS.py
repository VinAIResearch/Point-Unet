
import nibabel
import numpy as np
import concurrent.futures
import random
from os.path import join
import os, glob
import SimpleITK as sitk
import pickle
from scipy import ndimage
import copy
from tqdm import tqdm
import time
import cv2
import nibabel as nib
import SimpleITK as sitk


def save_to_nii(im, filename, outdir="", mode="image", system="nibabel"):
    """
    Save numpy array to nii.gz format to submit
    im: 3d numpy array ex: [155, 240, 240]
    """
    if system == "sitk":
        if mode == 'label':
            img = sitk.GetImageFromArray(im.astype(np.uint8))
        else:
            img = sitk.GetImageFromArray(im.astype(np.float32))
        if not os.path.exists("./{}".format(outdir)):
            os.mkdir("./{}".format(outdir))
        sitk.WriteImage(img, "./{}/{}.nii.gz".format(outdir, filename))
    elif system == "nibabel":
        # img = np.rot90(im, k=2, axes= (1,2))
        # print("img 0 ", im.shape)
        img = np.moveaxis(im, 0, -1)
        # print("img 1 ", img.shape)
        img = np.moveaxis(img, 0, 1)
        # print("img 2 ", img.shape)
        OUTPUT_AFFINE = np.array(
                [[ -1,-0,-0,-0],
                 [ -0,-1,-0,239],
                 [  0, 0, 1,0],
                 [  0, 0, 0,1]])
        if mode == 'label':
            img = nibabel.Nifti1Image(img.astype(np.uint8), OUTPUT_AFFINE)
        else:
            img = nibabel.Nifti1Image(img.astype(np.float32), OUTPUT_AFFINE)
        if not os.path.exists(outdir):
            os.mkdir(outdir)
        nibabel.save(img, "{}/{}.nii.gz".format(outdir, filename))
    else:
        img = np.rot90(im, k=2, axes= (1,2))
        OUTPUT_AFFINE = np.array(
                [[0, 0, 1, 0],
                [0, 1, 0, 0],
                [1, 0, 0, 0],
                [0, 0, 0, 1]])
        if mode == 'label':
            img = nibabel.Nifti1Image(img.astype(np.uint8), OUTPUT_AFFINE)
        else:
            img = nibabel.Nifti1Image(img.astype(np.float32), OUTPUT_AFFINE)
        if not os.path.exists("./{}".format(outdir)):
            os.mkdir("./{}".format(outdir))
        nibabel.save(img, "./{}/{}.nii.gz".format(outdir, filename))


def genSegmentation(pairInOut):
    
    in_pred = pairInOut[0]
    pathSave3D = pairInOut[1]
    ID_submit = in_pred.split('/')[-1].split('.npy')[0]
    print("Start with: ",ID_submit)
    prod_result = np.load(in_pred)
    seg = np.argmax(prod_result, axis=-1)
    seg = seg.astype(np.uint8)
    seg[seg == 3] = 4
    save_to_nii(seg,ID_submit,pathSave3D)
    print("Done with: ",ID_submit)


if __name__ == '__main__':

    pathProb = "/home/ubuntu/Research/3D_Med_Seg/Point-Unet/dataset/BraTS2020/predict_npy/"
    path3DVolume = "/home/ubuntu/Research/3D_Med_Seg/Point-Unet/dataset/BraTS2020/predict_nii"
    if not os.path.exists(path3DVolume):
        os.makedirs(path3DVolume)
    parallel = False

    listIDs = os.listdir(pathProb)
    listPairInOut = []
    for ID in listIDs:
        pathProbID = os.path.join(pathProb,ID)
        pairInOut = [pathProbID, path3DVolume]
        listPairInOut.append(pairInOut)

    if parallel:
        print("run with parallel")
        with concurrent.futures.ProcessPoolExecutor(50) as executor:
            tqdm(executor.map(genSegmentation, listPairInOut), total=len(listPairInOut))

    else:
        for pairInOutID in tqdm(listPairInOut):
            genSegmentation(pairInOutID)
