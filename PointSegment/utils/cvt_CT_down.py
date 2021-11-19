"""
获取可用于训练网络的训练数据集
需要四十分钟左右,产生的训练数据大小3G左右
"""

import os
import sys
sys.path.append(os.path.split(sys.path[0])[0])
import shutil
from time import time

import numpy as np
from tqdm import tqdm
import SimpleITK as sitk
import scipy.ndimage as ndimage

#Path to store processed data
training_set_path = '/home/ubuntu/Research/dataset/Pancreas-CT_processed_down_scale0.5_expand20'
#Path of original data
train_ct_path = '/home/ubuntu/NIH-Pancreas-CT/data/'
train_seg_path = '/home/ubuntu/NIH-Pancreas-CT/TCIA_pancreas_labels-02-05-2017'

#Maximum value
upper = 240
lower = -100

#Downsampling scale for x and y
down_scale = 0.5

slice_thickness = 1

expand_slice = 20

if os.path.exists(training_set_path):
    shutil.rmtree(training_set_path)

new_ct_path = os.path.join(training_set_path, 'ct')
new_seg_dir = os.path.join(training_set_path, 'seg')

os.mkdir(training_set_path)
os.mkdir(new_ct_path)
os.mkdir(new_seg_dir)

start_slices = [43, 151, 167]
end_slices = [227, 368, 405]

FULL_SIZE=True

if not FULL_SIZE:
    for i in range(3):
        start_slices[i] = start_slices[i] - expand_slice
        end_slices[i] = end_slices[i] + expand_slice

# mean_z = []
# mean_y = []
# mean_x = []

start = time()
for file in tqdm(os.listdir(train_ct_path)):

    # 将CT和金标准入读内存
    print(os.path.join(train_ct_path, file))
    ct = sitk.ReadImage(os.path.join(train_ct_path, file), sitk.sitkInt16)
    ct_array = sitk.GetArrayFromImage(ct)
    # print(ct.GetSpacing())
    # print(ct_array.shape)
    # print(ct.GetDirection())
    # print(ct.GetOrigin())

    seg = sitk.ReadImage(os.path.join(train_seg_path, file.replace('PANCREAS_', 'label')), sitk.sitkUInt8)
    seg_array = sitk.GetArrayFromImage(seg)
    # print(seg.GetSpacing())
    # print(seg.GetDirection())
    # print(seg.GetOrigin())

    # # 将金标准中肝脏和肝肿瘤的标签融合为一个
    # seg_array[seg_array > 0] = 1

    if ct.GetSpacing()[-1] != slice_thickness:
        ct_array = ndimage.zoom(ct_array, (ct.GetSpacing()[-1] / slice_thickness, 1, 1), order=3)
        # print(ct_array.shape)
        seg_array = ndimage.zoom(seg_array, (ct.GetSpacing()[-1] / slice_thickness, 1, 1), order=0)
        # print(seg_array.shape)

    if not FULL_SIZE:
        for i in range(3):
            start_slices[i] = max(0, start_slices[i])
            end_slices[i] = min(seg_array.shape[i] - 1, end_slices[i])

        ct_array = ct_array[start_slices[0]:end_slices[0] + 1, start_slices[1]:end_slices[1] + 1, start_slices[2]:end_slices[2] + 1]
        #The dataset mismatch between label and data
        ct_array = np.flip(ct_array, 1)
        seg_array = seg_array[start_slices[0]:end_slices[0] + 1, start_slices[1]:end_slices[1] + 1, start_slices[2]:end_slices[2] + 1]

    # 对CT数据在横断面上进行降采样,并进行重采样,将所有数据的z轴的spacing调整到1mm
    if down_scale != 1:
        ct_array = ndimage.zoom(ct_array, (down_scale, down_scale, down_scale), order=3)
        # print(ct_array.shape)
        seg_array = ndimage.zoom(seg_array, (down_scale, down_scale, down_scale), order=0)
        # print(seg_array.shape)

    # 将灰度值在阈值之外的截断掉
    ct_array[ct_array > upper] = upper
    ct_array[ct_array < lower] = lower

    # if ct_array.shape[0] < min_z:
    #     min_z = ct_array.shape[0]
    # elif ct_array.shape[0] > max_z:
    #     max_z = ct_array.shape[0]

    # 找到肝脏区域开始和结束的slice，并各向外扩张slice
    # z = np.any(seg_array, axis=(1, 2))
    # x = np.any(seg_array, axis=(0,1))
    # y = np.any(seg_array, axis=(0, 2))
    # mean_z.append(np.where(z)[0][[-1]] - np.where(z)[0][[0]])
    # mean_x.append(np.where(x)[0][[-1]] - np.where(x)[0][[0]])
    # mean_y.append(np.where(y)[0][[-1]] - np.where(y)[0][[0]])
    # mean_z.append(np.where(z)[0][[-1]])
    # mean_x.append(np.where(x)[0][[-1]])
    # mean_y.append(np.where(y)[0][[-1]])
    # mean_z.append(np.where(z)[0][[0]])
    # mean_x.append(np.where(x)[0][[0]])
    # mean_y.append(np.where(y)[0][[0]])
    # print(np.where(z)[0][[0]] - np.where(z)[0][[-1]])
    # print(np.where(x)[0][[0]] - np.where(x)[0][[-1]])
    # print(np.where(y)[0][[0]] - np.where(y)[0][[-1]])
    # start_slice, end_slice = np.where(z)[0][[0, -1]]

    # 两个方向上各扩张slice
    # start_slice = max(0, start_slice - expand_slice)
    # end_slice = min(seg_array.shape[0] - 1, end_slice + expand_slice)

    # # # 如果这时候剩下的slice数量不足size，直接放弃该数据，这样的数据很少,所以不用担心
    # # if end_slice - start_slice + 1 < para.size:
    # #     print('!!!!!!!!!!!!!!!!')
    # #     print(file, 'have too little slice', ct_array.shape[0])
    # #     print('!!!!!!!!!!!!!!!!')
    # #     continue

    print(ct_array.shape)
    print(seg_array.shape)

    # 最终将数据保存为nii
    new_ct = sitk.GetImageFromArray(ct_array)

    new_ct.SetDirection(ct.GetDirection())
    new_ct.SetOrigin(ct.GetOrigin())
    new_ct.SetSpacing((ct.GetSpacing()[0] * int(1 / down_scale), ct.GetSpacing()[1] * int(1 / down_scale), slice_thickness / down_scale))

    new_seg = sitk.GetImageFromArray(seg_array)

    new_seg.SetDirection(ct.GetDirection())
    new_seg.SetOrigin(ct.GetOrigin())
    new_seg.SetSpacing((ct.GetSpacing()[0] * int(1 / down_scale), ct.GetSpacing()[1] * int(1 / down_scale), slice_thickness / down_scale))

    sitk.WriteImage(new_ct, os.path.join(new_ct_path, file))
    sitk.WriteImage(new_seg, os.path.join(new_seg_dir, file.replace('PANCREAS_', 'label')))
# print(min_z, max_z)
# print(np.max(mean_z), np.min(mean_z))
# print(np.max(mean_y), np.min(mean_y))
# print(np.max(mean_x), np.min(mean_x))