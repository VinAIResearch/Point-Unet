# -*- coding: utf-8 -*-
# File: eval.py

import tqdm
import os
from collections import namedtuple
import numpy as np
#import cv2
import time
import scipy.ndimage.morphology
from skimage import measure, filters

from tensorpack.utils.utils import get_tqdm_kwargs
import config
from utils import *
import nibabel as nib
import SimpleITK as sitk
from scipy.special import softmax

def post_processing(pred1, temp_weight):
    struct = ndimage.generate_binary_structure(3, 2)
    margin = 5
    wt_threshold = 2000
    pred1 = pred1 * temp_weight # clear non-brain region
    # pred1 should be the same as cropped brain region
    # now fill the croped region with our prediction
    pred_whole = np.zeros_like(pred1)
    pred_core = np.zeros_like(pred1)
    pred_enhancing = np.zeros_like(pred1)
    pred_whole[pred1 > 0] = 1
    pred1[pred1 == 2] = 0
    pred_core[pred1 > 0] = 1
    pred_enhancing[pred1 == 4]  = 1
    
    pred_whole = ndimage.morphology.binary_closing(pred_whole, structure = struct)
    pred_whole = get_largest_two_component(pred_whole, False, wt_threshold)
    
    sub_weight = np.zeros_like(temp_weight)
    sub_weight[pred_whole > 0] = 1
    pred_core = pred_core * sub_weight
    pred_core = ndimage.morphology.binary_closing(pred_core, structure = struct)
    pred_core = get_largest_two_component(pred_core, False, wt_threshold)

    subsub_weight = np.zeros_like(temp_weight)
    subsub_weight[pred_core > 0] = 1
    pred_enhancing = pred_enhancing * subsub_weight
    vox_3  = np.asarray(pred_enhancing > 0, np.float32).sum()
    all_vox = np.asarray(pred_whole > 0, np.float32).sum()
    if(all_vox > 100 and 0 < vox_3 and vox_3 < 100):
        pred_enhancing = np.zeros_like(pred_enhancing)
    out_label = pred_whole * 2
    out_label[pred_core>0] = 1
    out_label[pred_enhancing>0] = 4

    return out_label

def batch_segmentation(temp_imgs, model_func, data_shape=[19, 180, 160]):
    batch_size = config.BATCH_SIZE
    data_channel = 4
    class_num = config.NUM_CLASS
    image_shape = temp_imgs[0].shape
    label_shape = [data_shape[0], data_shape[1], data_shape[2]]
    D, H, W = image_shape
    input_center = [int(D/2), int(H/2), int(W/2)]
    temp_prob1 = np.zeros([D, H, W, class_num])

    sub_image_batches = []
    for center_slice in range(int(label_shape[0]/2), D + int(label_shape[0]/2), label_shape[0]):
        center_slice = min(center_slice, D - int(label_shape[0]/2))
        sub_image_batch = []
        for chn in range(data_channel):
            temp_input_center = [center_slice, input_center[1], input_center[2]]
            sub_image = extract_roi_from_volume(
                            temp_imgs[chn], temp_input_center, data_shape, fill="zero")
            sub_image_batch.append(sub_image)
        sub_image_batch = np.asanyarray(sub_image_batch, np.float32) #[4,180,160]
        sub_image_batches.append(sub_image_batch) # [14,4,d,h,w]
    
    total_batch = len(sub_image_batches)
    max_mini_batch = int((total_batch+batch_size-1)/batch_size)
    sub_label_idx1 = 0
    for mini_batch_idx in range(max_mini_batch):
        data_mini_batch = sub_image_batches[mini_batch_idx*batch_size:
                                      min((mini_batch_idx+1)*batch_size, total_batch)]
        if(mini_batch_idx == max_mini_batch - 1):
            for idx in range(batch_size - (total_batch - mini_batch_idx*batch_size)):
                data_mini_batch.append(np.zeros([data_channel] + data_shape))
                # data_mini_batch.append(np.random.normal(0, 1, size = [data_channel] + data_shape))
        data_mini_batch = np.asarray(data_mini_batch, np.float32)
        data_mini_batch = np.transpose(data_mini_batch, [0, 2, 3, 4, 1])
        prob_mini_batch1, _ = model_func(data_mini_batch)
        
        for batch_idx in range(prob_mini_batch1.shape[0]):
            center_slice = sub_label_idx1*label_shape[0] + int(label_shape[0]/2)
            center_slice = min(center_slice, D - int(label_shape[0]/2))
            temp_input_center = [center_slice, input_center[1], input_center[2], int(class_num/2)]
            sub_prob = np.reshape(prob_mini_batch1[batch_idx], label_shape + [class_num])
            temp_prob1 = set_roi_to_volume(temp_prob1, temp_input_center, sub_prob)
            sub_label_idx1 = sub_label_idx1 + 1
    
    return temp_prob1

def overlapping_inference(temp_imgs, model_func, data_shape):
    start = time.time()
    crop_size = data_shape

    #full size
    xstep = 48
    ystep = zstep = 118
    # xstep = 32
    # ystep = zstep = 51

    # #Margin 20
    # xstep = 32
    # ystep = zstep = 40

    # #Margin 40
    # xstep = 48
    # ystep = zstep = 81
    # xstep = 40
    # ystep = zstep = 54

    # downsample full size
    # xstep = 16
    # ystep = zstep = 32

    # downsample margin 20
    # xstep = 6
    # ystep = zstep = 1

    # downsample margin 40
    # xstep = 10
    # ystep = zstep = 1

    image = temp_imgs
    image = np.array(image)
    image = np.rollaxis(image, 0, 4)
    image = np.expand_dims(image, 0)
    #print(image.shape)

    _, D, H, W, _ = image.shape
    deep_slices   = np.arange(0, max(1, D - crop_size[0] + xstep), xstep)
    height_slices = np.arange(0, max(1, H - crop_size[1] + ystep), ystep)
    width_slices  = np.arange(0, max(1, W - crop_size[2] + zstep), zstep)
    # print(len(deep_slices))
    # print(len(height_slices))
    # print(len(width_slices))

    whole_pred = np.zeros(image.shape[:-1] + (config.NUM_CLASS,))
    #print(whole_pred.shape)
    count_used = np.zeros((D, H, W))

    for j in range(len(deep_slices)):
        for k in range(len(height_slices)):
            for l in range(len(width_slices)):
                deep = deep_slices[j]
                height = height_slices[k]
                width = width_slices[l]
                image_input = np.zeros(shape = (config.BATCH_SIZE,) + tuple(data_shape) + (1,))
                image_crop = image[:, deep   : deep   + crop_size[0],
                                            height : height + crop_size[1],
                                            width  : width  + crop_size[2], :]
                image_input[:, :image_crop.shape[1], :image_crop.shape[2], :image_crop.shape[3], :] = image_crop

                pred, _ = model_func(image_input)
                #print(outputs[0].shape)
                #----------------Average-------------------------------
                whole_pred[:, deep: deep + crop_size[0],
                            height: height + crop_size[1],
                            width: width + crop_size[2], :] += pred[0, :image_crop.shape[1], :image_crop.shape[2], :image_crop.shape[3], :]

                count_used[deep: deep + crop_size[0],
                            height: height + crop_size[1],
                            width: width + crop_size[2]] += 1
    #print(whole_pred.shape)
    count_used = np.expand_dims(count_used, (0, -1))
    whole_pred = whole_pred / count_used
    #whole_pred = softmax(whole_pred, axis=-1)
    # final_pred = np.argmax(whole_pred, axis=-1)
    # #print(whole_pred.shape)

    # affine = np.array([
    #         [-1., -0., -0., 0.],
    #         [-0., -1., -0., 239.],
    #         [ 0., 0., 1., 0.],
    #         [ 0., 0., 0., 1.],
    #     ])

    # img = nib.Nifti1Image(final_pred, affine)
    # nib.save(img, os.path.join('../output', 'test.nii.gz'))  
    # print(f"writing to test.nii.gz: {time.time()-start:.02f} sec")

    return np.squeeze(whole_pred)

def segment_one_image_dynamic(data, create_model_func):
    """
    Change PATCH_SIZE in inference if cropped brain region > PATCH_SIZE
    NOTE: After testing, this function makes little difference 
            compared to setting larger patch_size at first place.
    """
    def get_dynamic_shape(image_shape):
        [D, H, W] = image_shape
        data_shape = config.INFERENCE_PATCH_SIZE
        Hx = max(int((H+3)/4)*4, data_shape[1])
        Wx = max(int((W+3)/4)*4, data_shape[2])
        data_slice = data_shape[0]
        label_slice = data_shape[0]
        full_data_shape = [data_slice, Hx, Wx]
        return full_data_shape

    img = data['images']
    temp_weight = data['weights'][:,:,:,0]
    temp_size = data['original_shape']
    temp_bbox = data['bbox']
    
    img = img[np.newaxis, ...] # add batch dim

    im = img
    
    if config.MULTI_VIEW:
        im_ax = np.transpose(im[0], [3, 0 ,1, 2]) # mod, d, h, w
        im_ax = transpose_volumes(im_ax, 'axial')
        [D, H, W] = im_ax.shape
        if not (H <= config.INFERENCE_PATCH_SIZE[1] and W <= config.INFERENCE_PATCH_SIZE[2]):
            full_data_shape = get_dynamic_shape(im_ax[0].shape)
            dy_model_func = create_model_func[0](full_data_shape)
            prob1_ax = batch_segmentation(im_ax, dy_model_func, data_shape=full_data_shape)
        else:
            dy_model_func = create_model_func[0](config.INFERENCE_PATCH_SIZE)
            prob1_ax = batch_segmentation(im_ax, dy_model_func, data_shape=config.INFERENCE_PATCH_SIZE)
   
        im_sa = np.transpose(im[0], [3, 0 ,1, 2]) # mod, d, h, w
        im_sa = transpose_volumes(im_sa, 'sagittal')
        [D, H, W] = im_sa.shape
        if not (H <= config.INFERENCE_PATCH_SIZE[1] and W <= config.INFERENCE_PATCH_SIZE[2]):
            full_data_shape = get_dynamic_shape(im_sa.shape)
            dy_model_func = create_model_func[1](full_data_shape)
            prob1_sa = batch_segmentation(im_sa, dy_model_func, data_shape=full_data_shape)
        else:
            dy_model_func = create_model_func[1](config.INFERENCE_PATCH_SIZE)
            prob1_sa = batch_segmentation(im_sa, dy_model_func, data_shape=config.INFERENCE_PATCH_SIZE)

        im_co = np.transpose(im[0], [3, 0 ,1, 2]) # mod, d, h, w
        im_co = transpose_volumes(im_co, 'coronal')
        [D, H, W] = im_co.shape
        if not (H <= config.INFERENCE_PATCH_SIZE[1] and W <= config.INFERENCE_PATCH_SIZE[2]):
            full_data_shape = get_dynamic_shape(im_co.shape)
            dy_model_func = create_model_func[2](full_data_shape)
            prob1_co = batch_segmentation(im_co, dy_model_func, data_shape=full_data_shape)
        else:
            dy_model_func = create_model_func[2](config.INFERENCE_PATCH_SIZE)
            prob1_co = batch_segmentation(im_co, dy_model_func, data_shape=config.INFERENCE_PATCH_SIZE)

        prob1 = (prob1_ax + np.transpose(prob1_sa, (1,2,0,3)) + np.transpose(prob1_co, (1,0,2,3)))/ 3.0
        pred1 = np.argmax(prob1, axis=-1)
    else:
        im_ax = np.transpose(im[0], [3, 0 ,1, 2]) # mod, d, h, w
        im_ax = transpose_volumes(im_ax, config.DIRECTION)
        [D, H, W] = im_ax.shape
        if not (H <= config.INFERENCE_PATCH_SIZE[1] and W <= config.INFERENCE_PATCH_SIZE[2]):
            full_data_shape = get_dynamic_shape(im_ax[0].shape)
            dy_model_func = create_model_func[0](full_data_shape)
            prob1 = batch_segmentation(im_ax, dy_model_func, data_shape=full_data_shape)
        else:
            dy_model_func = create_model_func[0](config.INFERENCE_PATCH_SIZE)
            prob1 = batch_segmentation(im_ax, dy_model_func, data_shape=config.INFERENCE_PATCH_SIZE)
        # need to take care if image size > data_shape

        pred1 = np.argmax(prob1, axis=-1)
        
    pred1[pred1 == 3] = 4
    out_label = post_processing(pred1, temp_weight)
    out_label = np.asarray(out_label, np.int16)
    if 'is_flipped' in data and data['is_flipped']:
        out_label = np.flip(out_label, axis=-1)
        prob1 = np.flip(prob1, axis=2) # d, h, w, num_class

    final_label = np.zeros(temp_size, np.int16)
    final_label = set_ND_volume_roi_with_bounding_box_range(final_label, temp_bbox[0], temp_bbox[1], out_label)
    
    final_probs = np.zeros(list(temp_size) + [config.NUM_CLASS], np.float32)
    final_probs = set_ND_volume_roi_with_bounding_box_range(final_probs, temp_bbox[0]+[0], temp_bbox[1]+[3], prob1)
        
    return final_label, final_probs

# def segment_one_image(data, model_func, is_online=False):
#     """
#     perform inference and unpad the volume to original shape
#     """
#     img = data['images']
#     temp_weight = data['weights'][:,:,:,0]
#     temp_size = data['original_shape']
#     temp_bbox = data['bbox']
#     # Ensure online evaluation match the training patch shape...should change in future 
#     batch_data_shape = config.PATCH_SIZE if is_online else config.INFERENCE_PATCH_SIZE
    
#     img = img[np.newaxis, ...] # add batch dim

#     im = img

#     if config.MULTI_VIEW:
#         im_ax = np.transpose(im[0], [3, 0 ,1, 2]) # mod, d, h, w
#         im_ax = transpose_volumes(im_ax, 'axial')
#         prob1_ax = batch_segmentation(im_ax, model_func[0], data_shape=batch_data_shape)

#         im_sa = np.transpose(im[0], [3, 0 ,1, 2]) # mod, d, h, w
#         im_sa = transpose_volumes(im_sa, 'sagittal')
#         prob1_sa = batch_segmentation(im_sa, model_func[1], data_shape=batch_data_shape)

#         im_co = np.transpose(im[0], [3, 0 ,1, 2]) # mod, d, h, w
#         im_co = transpose_volumes(im_co, 'coronal')
#         prob1_co = batch_segmentation(im_co, model_func[2], data_shape=batch_data_shape)

#         prob1 = (prob1_ax + np.transpose(prob1_sa, (1, 2, 0, 3)) + np.transpose(prob1_co, (1, 0, 2, 3))) / 3.0
#         pred1 = np.argmax(prob1, axis=-1)
        
#     else:
#         im_pred = np.transpose(im[0], [3, 0 ,1, 2]) # mod, d, h, w
#         im_pred = transpose_volumes(im_pred, config.DIRECTION)
#         # prob1 = batch_segmentation(im_pred, model_func[0], data_shape=batch_data_shape)
#         prob1 = overlapping_inference(im_pred, model_func[0], data_shape=batch_data_shape)
#         if config.DIRECTION == 'sagittal':
#             prob1 = np.transpose(prob1, (1, 2, 0, 3))
#         elif config.DIRECTION == 'coronal':
#             prob1 = np.transpose(prob1, (1, 0, 2, 3))
#         else:
#             prob1 = prob1
        
#         if config.NUM_CLASS == 1:
#             pred1 = prob1 >= 0.5
#             pred1 = np.squeeze(pred1, axis=-1)
#         else:
#             pred1 = np.argmax(prob1, axis=-1)
    
#     pred1[pred1 == 3] = 4
#     # pred1 should be the same as cropped brain region
#     if config.ADVANCE_POSTPROCESSING:
#         out_label = post_processing(pred1, temp_weight)
#     else:
#         out_label = pred1
#     out_label = np.asarray(out_label, np.int16)

#     if 'is_flipped' in data and data['is_flipped']:
#         out_label = np.flip(out_label, axis=-1)
#         prob1 = np.flip(prob1, axis=2) # d, h, w, num_class
    
#     final_label = np.zeros(temp_size, np.int16)
#     final_label = set_ND_volume_roi_with_bounding_box_range(final_label, temp_bbox[0], temp_bbox[1], out_label)

#     final_probs = np.zeros(list(temp_size) + [config.NUM_CLASS], np.float32)
#     final_probs = set_ND_volume_roi_with_bounding_box_range(final_probs, temp_bbox[0]+[0], temp_bbox[1]+[config.NUM_CLASS - 1], prob1)
        
#     return final_label, final_probs

def segment_one_image(data, model_func, is_online=False):
    """
    perform inference and unpad the volume to original shape
    """
    img = data['images']
    temp_weight = data['weights'][:,:,:,0]
    # Ensure online evaluation match the training patch shape...should change in future 
    batch_data_shape = config.PATCH_SIZE if is_online else config.INFERENCE_PATCH_SIZE
    
    img = img[np.newaxis, ...] # add batch dim

    im = img

    im_pred = np.transpose(im[0], [3, 0 ,1, 2]) # mod, d, h, w
    im_pred = transpose_volumes(im_pred, config.DIRECTION)
    # prob1 = batch_segmentation(im_pred, model_func[0], data_shape=batch_data_shape)
    prob1 = overlapping_inference(im_pred, model_func[0], data_shape=batch_data_shape)
    if config.DIRECTION == 'sagittal':
        prob1 = np.transpose(prob1, (1, 2, 0, 3))
    elif config.DIRECTION == 'coronal':
        prob1 = np.transpose(prob1, (1, 0, 2, 3))
    else:
        prob1 = prob1
    
    if config.NUM_CLASS == 1:
        pred1 = prob1 >= 0.5
        pred1 = np.squeeze(pred1, axis=-1)
    else:
        # prob1[:,:,:,1] = prob1[:,:,:,1] + 0.2
        pred1 = np.argmax(prob1, axis=-1)

    # pred1 should be the same as cropped brain region
    if config.ADVANCE_POSTPROCESSING:
        # all_labels = measure.label(pred1, connectivity=3)
        # props = measure.regionprops(all_labels)
        # props.sort(key=lambda x: x.area, reverse=True)
        # thresholded_mask = np.zeros(pred1.shape)

        # if len(props) >= 2:
        #     if props[0].area / props[1].area > 5:  # if the largest is way larger than the second largest
        #         thresholded_mask[all_labels == props[0].label] = 1  # only turn on the largest component
        #     else:
        #         thresholded_mask[all_labels == props[0].label] = 1  # turn on two largest components
        #         thresholded_mask[all_labels == props[1].label] = 1
        # elif len(props):
        #     thresholded_mask[all_labels == props[0].label] = 1

        out_label = scipy.ndimage.morphology.binary_fill_holes(pred1).astype(np.uint8)
    else:
        out_label = pred1
    out_label = np.asarray(out_label, np.int16)

    if 'is_flipped' in data and data['is_flipped']:
        out_label = np.flip(out_label, axis=-1)
        prob1 = np.flip(prob1, axis=2) # d, h, w, num_class
        
    return out_label, prob1

def dice_of_brats_data_set(gt, pred, type_idx):
    dice_all_data = []
    for i in range(len(gt)):
        g_volume = gt[i]
        s_volume = pred[i]
        dice_one_volume = []
        if(type_idx ==0): # whole tumor
            if config.NUM_CLASS == 2:
                g_volume[g_volume == 4] = 1
                g_volume[g_volume == 2] = 1
            temp_dice = binary_dice3d(s_volume > 0, g_volume > 0)
            dice_one_volume = [temp_dice]
        elif(type_idx == 1): # tumor core
            s_volume[s_volume == 2] = 0
            g_volume[g_volume == 2] = 0
            temp_dice = binary_dice3d(s_volume > 0, g_volume > 0)
            dice_one_volume = [temp_dice]
        else:
            #for label in [1, 2, 3, 4]: # dice of each class
            temp_dice = binary_dice3d(s_volume == 4, g_volume == 4)
            dice_one_volume = [temp_dice]
        dice_all_data.append(dice_one_volume)
    return dice_all_data

def dice_of_pancreas(gt, pred, type_idx):
    dice_all_data = []
    for i in range(len(gt)):
        g_volume = gt[i]
        s_volume = pred[i]
        dice_one_volume = []
        if(type_idx ==0): # whole tumor
            temp_dice = binary_dice3d(s_volume, g_volume)
            dice_one_volume = [temp_dice]
        else:
            pass
        dice_all_data.append(dice_one_volume)
    return dice_all_data

def eval_brats(df, detect_func, with_gt=True):
    """
    evalutation
    """
    df.reset_state()
    gts = []
    results = []
    with tqdm.tqdm(total=df.size(), **get_tqdm_kwargs()) as pbar:
        for filename, image_id, data in df.get_data():
            final_label, probs = detect_func(data)
            if config.TEST_FLIP:
                pred_flip, probs_flip = detect_func(flip_lr(data))
                final_prob = (probs + probs_flip) / 2.0
                pred = np.argmax(final_prob, axis=-1)
                pred[pred == 3] = 4
                if config.ADVANCE_POSTPROCESSING:
                    pred = crop_ND_volume_with_bounding_box(pred, data['bbox'][0], data['bbox'][1])
                    pred = post_processing(pred, data['weights'][:,:,:,0])
                    pred = np.asarray(pred, np.int16)
                    final_label = np.zeros(data['original_shape'], np.int16)
                    final_label = set_ND_volume_roi_with_bounding_box_range(final_label, data['bbox'][0], data['bbox'][1], pred)
                else:
                    final_label = pred
            gt = load_nifty_volume_as_array("{}/{}_seg.nii.gz".format(filename, image_id))
            gts.append(gt)
            results.append(final_label)
            pbar.update()
    test_types = ['whole', 'core', 'enhancing']
    ret = {}
    class_num = config.NUM_CLASS if config.NUM_CLASS == 1 else config.NUM_CLASS - 1
    for type_idx in range(class_num):
        dice = dice_of_brats_data_set(gts, results, type_idx)
        dice = np.asarray(dice)
        dice_mean = dice.mean(axis = 0)
        dice_std  = dice.std(axis = 0)
        test_type = test_types[type_idx]
        ret[test_type] = dice_mean[0]
        print('tissue type', test_type)
        print('dice mean', dice_mean)
    return ret

def eval_pancreas(df, detect_func, with_gt=True):
    """
    evalutation
    """
    df.reset_state()
    gts = []
    results = []
    if not os.path.isdir(os.path.join(config.BASEDIR, 'probs')):
        os.mkdir(os.path.join(config.BASEDIR, 'probs'))
    if not os.path.isdir(os.path.join(config.BASEDIR, 'predictions')):
        os.mkdir(os.path.join(config.BASEDIR, 'predictions'))
    with tqdm.tqdm(total=df.size(), **get_tqdm_kwargs()) as pbar:
        for filename, image_id, data in df.get_data():
            probs_path = os.path.join(config.BASEDIR, 'probs', filename.split('/')[-1].replace('nii.gz', 'npy'))
            # print(probs_path)
            final_label, probs = detect_func(data)

            # ct = sitk.ReadImage(filename, sitk.sitkInt16)
            # new_seg = sitk.GetImageFromArray(final_label)

            # new_seg.SetDirection(ct.GetDirection())
            # new_seg.SetOrigin(ct.GetOrigin())
            # new_seg.SetSpacing((ct.GetSpacing()[0], ct.GetSpacing()[1], ct.GetSpacing()[2]))

            # sitk.WriteImage(new_seg, os.path.join(config.BASEDIR, 'predictions', filename.split('/')[-1].replace('PANCREAS_', 'label')))
            # np.save(probs_path, probs)
            if config.TEST_FLIP:
                pred_flip, probs_flip = detect_func(flip_lr(data))
                final_prob = (probs + probs_flip) / 2.0
                pred = np.argmax(final_prob, axis=-1)
                pred[pred == 3] = 4
                if config.ADVANCE_POSTPROCESSING:
                    pred = crop_ND_volume_with_bounding_box(pred, data['bbox'][0], data['bbox'][1])
                    pred = post_processing(pred, data['weights'][:,:,:,0])
                    pred = np.asarray(pred, np.int16)
                    final_label = np.zeros(data['original_shape'], np.int16)
                    final_label = set_ND_volume_roi_with_bounding_box_range(final_label, data['bbox'][0], data['bbox'][1], pred)
                else:
                    final_label = pred
            gt = np.squeeze(data['labels'])
            gts.append(gt)
            results.append(final_label)
            pbar.update()
    test_types = ['pancreas']
    ret = {}
    class_num = config.NUM_CLASS if config.NUM_CLASS == 1 else config.NUM_CLASS - 1
    for type_idx in range(class_num):
        dice = dice_of_pancreas(gts, results, type_idx)
        dice = np.asarray(dice)
        for score in dice:
            print(score)
        dice_mean = dice.mean(axis = 0)
        dice_std  = dice.std(axis = 0)
        test_type = test_types[type_idx]
        ret[test_type] = dice_mean[0]
        print('tissue type', test_type)
        print('dice mean', dice_mean)
    return ret

def pred_brats(df, detect_func):
    """
    prediction
    """

    df.reset_state()

    gts = []
    results = []



    with tqdm.tqdm(total=df.size(), **get_tqdm_kwargs()) as pbar:
        for filename, image_id, data in df.get_data():

            final_label, probs = detect_func(data)
            # pred = np.argmax(probs, axis=-1)
            # if (final_label== pred).all() == pred.all():
            #     print("Yeah yeah")
            # print("final_label ", final_label.shape)
            #print("probs ", probs.shape)
            # probs_path = '../probs/BraTS2018_val/' + image_id + '.npy'
            # print(probs_path)
            # np.save(probs_path, probs)
            if config.TEST_FLIP:
                pred_flip, probs_flip = detect_func(flip_lr(data))
                final_prob = (probs + probs_flip) / 2.0

                # probs_path = '../probs/BraTS2020_val/' + image_id + '.npy'
                # print(probs_path)
                # np.save(probs_path, final_prob)

                pred = np.argmax(final_prob, axis=-1)
                pred[pred == 3] = 4
                if config.ADVANCE_POSTPROCESSING:
                    pred = crop_ND_volume_with_bounding_box(pred, data['bbox'][0], data['bbox'][1])
                    pred = post_processing(pred, data['weights'][:,:,:,0])
                    pred = np.asarray(pred, np.int16)
                    final_label = np.zeros(data['original_shape'], np.int16)
                    final_label = set_ND_volume_roi_with_bounding_box_range(final_label, data['bbox'][0], data['bbox'][1], pred)
                else:
                    final_label = pred
            #if np.count_nonzero(final_label == 4) < 620:
            #    final_label[final_label == 4]=0
            save_to_nii(final_label, image_id, outdir=config.save_pred, mode="label")
            # save prob to ensemble
            # save_to_pkl(probs, image_id, outdir="eval_out18_prob_{}".format(config.CROSS_VALIDATION))
            pbar.update()
    return None
 