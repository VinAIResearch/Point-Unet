#####
# some functions are borrowed from https://github.com/taigw/brats17/
#####

import nibabel
import numpy as np
import random
import os, sys, glob, pickle
from os.path import join, exists, dirname, abspath
# import SimpleITK as sitk
import pickle
from scipy import ndimage
import copy
from sklearn.model_selection import train_test_split
import matplotlib.image as plt
import concurrent.futures
from tqdm import tqdm
BASE_DIR = dirname(abspath(__file__))
ROOT_DIR = dirname(BASE_DIR)

sys.path.append(BASE_DIR)
sys.path.append(ROOT_DIR)
from helper_ply import write_ply

number_of_point = 262144
INTENSITY_NORM = 'modality'
PATCH_SIZE = (144,144,144)
DIRECTION = 'axial'
path_origin ='/home/ubuntu/Research/BraTS/BraTS_data/MICCAI_BraTS2020_TrainingData/training/HGG/'
output_savepoint = "/home/ubuntu/Point_Cloud/block64_xyz_local/val_set_x_z_y_local/"
def flip_lr(data):
    data = copy.deepcopy(data)
    img = data['images']
    img = np.transpose(img, [3, 0 ,1, 2])
    weight = data['weights'][:,:,:,0]
    flipped_data = []
    for moda in range(len(img)):
        flipped_data.append(np.flip(img[moda], axis=-1))
    flipped_data = np.array(flipped_data)
    weight = np.flip(weight, axis=-1)
    data['images'] = np.transpose(flipped_data, [1, 2, 3, 0])
    data['weights'] = np.transpose(weight[np.newaxis, ...], [1, 2, 3, 0])
    data['is_flipped'] = True
    return data

def crop_brain_region(im, gt, with_gt=True):
    mods = sorted(im.keys())
    volume_list = []
    weigh_final = None
    for mod_idx, mod in enumerate(mods):
        filename = im[mod]
        volume = load_nifty_volume_as_array(filename, with_header=False)
        # 155 244 244
        if mod_idx == 0:
            # contain whole tumor
            margin = 5 # small padding value
            original_shape = volume.shape
            bbmin, bbmax = get_none_zero_region(volume, margin)
        volume = crop_ND_volume_with_bounding_box(volume, bbmin, bbmax)
        # print("volume.shape ",volume.shape)
        if mod_idx == 0:
            weight = np.asarray(volume > 0, np.int8)
            weigh_final = weight
        else:
            weight = np.asarray(volume > 0, np.int8)
            weigh_final = np.bitwise_or(weigh_final, weight)
        if INTENSITY_NORM == 'modality':
            volume = itensity_normalize_one_volume(volume)
        volume_list.append(volume)
    ## volume_list [(depth, h, w)*4]
    if with_gt:
        label = load_nifty_volume_as_array(gt, False)
        label[label == 4] = 3
        label = crop_ND_volume_with_bounding_box(label, bbmin, bbmax)
        # print("label.shape ",label.shape)
        return volume_list, label.astype(int), weigh_final.astype(int), original_shape, [bbmin, bbmax]
    else:
        return volume_list, None, weigh_final, original_shape, [bbmin, bbmax]

def transpose_volumes(volumes, slice_direction):
    """
    transpose a list of volumes
    inputs:
        volumes: a list of nd volumes
        slice_direction: 'axial', 'sagittal', or 'coronal'
    outputs:
        tr_volumes: a list of transposed volumes
    """
    if (slice_direction == 'axial'):
        tr_volumes = volumes
    elif(slice_direction == 'sagittal'):
        if isinstance(volumes, list) or len(volumes.shape) > 3:
            tr_volumes = [np.transpose(x, (2, 0, 1)) for x in volumes]
        else:
            tr_volumes = np.transpose(volumes, (2, 0, 1))
    elif(slice_direction == 'coronal'):
        if isinstance(volumes, list) or len(volumes.shape) > 3:
            tr_volumes = [np.transpose(x, (1, 0, 2)) for x in volumes]
        else:
            tr_volumes = np.transpose(volumes, (1, 0, 2))
    else:
        print('undefined slice direction:', slice_direction)
        tr_volumes = volumes
    return tr_volumes

def remove_external_core(lab_main, lab_ext):
    """
    remove the core region that is outside of whole tumor
    """
    
    # for each component of lab_ext, compute the overlap with lab_main
    s = ndimage.generate_binary_structure(3,2) # iterate structure
    labeled_array, numpatches = ndimage.label(lab_ext,s) # labeling
    sizes = ndimage.sum(lab_ext,labeled_array,range(1,numpatches+1)) 
    sizes_list = [sizes[i] for i in range(len(sizes))]
    new_lab_ext = np.zeros_like(lab_ext)
    for i in range(len(sizes)):
        sizei = sizes_list[i]
        labeli =  np.where(sizes == sizei)[0] + 1
        componenti = labeled_array == labeli
        overlap = componenti * lab_main
        if((overlap.sum()+ 0.0)/sizei >= 0.5):
            new_lab_ext = np.maximum(new_lab_ext, componenti)
    return new_lab_ext


def get_largest_two_component(img, print_info = False, threshold = None):
    """
    Get the largest two components of a binary volume
    inputs:
        img: the input 3D volume
        threshold: a size threshold
    outputs:
        out_img: the output volume 
    """
    s = ndimage.generate_binary_structure(3,2) # iterate structure
    labeled_array, numpatches = ndimage.label(img,s) # labeling
    sizes = ndimage.sum(img,labeled_array,range(1,numpatches+1)) 
    sizes_list = [sizes[i] for i in range(len(sizes))]
    sizes_list.sort()
    if(print_info):
        print('component size', sizes_list)
    if(len(sizes) == 1):
        out_img = img
    else:
        if(threshold):
            out_img = np.zeros_like(img)
            for temp_size in sizes_list:
                if(temp_size > threshold):
                    temp_lab = np.where(sizes == temp_size)[0] + 1
                    temp_cmp = labeled_array == temp_lab
                    out_img = (out_img + temp_cmp) > 0
            return out_img
        else:    
            max_size1 = sizes_list[-1]
            max_size2 = sizes_list[-2]
            max_label1 = np.where(sizes == max_size1)[0] + 1
            max_label2 = np.where(sizes == max_size2)[0] + 1
            component1 = labeled_array == max_label1
            component2 = labeled_array == max_label2
            if(max_size2*10 > max_size1):
                component1 = (component1 + component2) > 0
            out_img = component1
    return out_img

def get_ND_bounding_box(label, margin):
    """
    get the bounding box of the non-zero region of an ND volume
    """
    input_shape = label.shape
    if(type(margin) is int ):
        margin = [margin]*len(input_shape)
    assert(len(input_shape) == len(margin))
    indxes = np.nonzero(label)
    idx_min = []
    idx_max = []
    for i in range(len(input_shape)):
        idx_min.append(indxes[i].min())
        idx_max.append(indxes[i].max())

    for i in range(len(input_shape)):
        idx_min[i] = max(idx_min[i] - margin[i], 0)
        idx_max[i] = min(idx_max[i] + margin[i], input_shape[i] - 1)
    return idx_min, idx_max

def set_ND_volume_roi_with_bounding_box_range(volume, bb_min, bb_max, sub_volume):
    """
    set a subregion to an nd image.
    """
    dim = len(bb_min)
    out = volume
    if(dim == 2):
        out[np.ix_(range(bb_min[0], bb_max[0] + 1),
                   range(bb_min[1], bb_max[1] + 1))] = sub_volume
    elif(dim == 3):
        out[np.ix_(range(bb_min[0], bb_max[0] + 1),
                   range(bb_min[1], bb_max[1] + 1),
                   range(bb_min[2], bb_max[2] + 1))] = sub_volume
    elif(dim == 4):
        out[np.ix_(range(bb_min[0], bb_max[0] + 1),
                   range(bb_min[1], bb_max[1] + 1),
                   range(bb_min[2], bb_max[2] + 1),
                   range(bb_min[3], bb_max[3] + 1))] = sub_volume
    else:
        raise ValueError("array dimension should be 2, 3 or 4")
    return out


def convert_label(in_volume, label_convert_source, label_convert_target):
    """
    convert the label value in a volume
    inputs:
        in_volume: input nd volume with label set label_convert_source
        label_convert_source: a list of integers denoting input labels, e.g., [0, 1, 2, 4]
        label_convert_target: a list of integers denoting output labels, e.g.,[0, 1, 2, 3]
    outputs:
        out_volume: the output nd volume with label set label_convert_target
    """
    mask_volume = np.zeros_like(in_volume)
    convert_volume = np.zeros_like(in_volume)
    for i in range(len(label_convert_source)):
        source_lab = label_convert_source[i]
        target_lab = label_convert_target[i]
        if(source_lab != target_lab):
            temp_source = np.asarray(in_volume == source_lab)
            temp_target = target_lab * temp_source
            mask_volume = mask_volume + temp_source
            convert_volume = convert_volume + temp_target
    out_volume = in_volume * 1
    out_volume[mask_volume>0] = convert_volume[mask_volume>0]
    return out_volume

def set_roi_to_volume(volume, center, sub_volume):
    """
    set the content of an roi of a 3d/4d volume to a sub volume
    inputs:
        volume: the input 3D/4D volume
        center: the center of the roi
        sub_volume: the content of sub volume
    outputs:
        output_volume: the output 3D/4D volume
    """
    volume_shape = volume.shape   
    patch_shape = sub_volume.shape
    output_volume = volume
    for i in range(len(center)):
        if(center[i] >= volume_shape[i]):
            return output_volume
    r0max = [int(x/2) for x in patch_shape]
    r1max = [patch_shape[i] - r0max[i] for i in range(len(r0max))]
    r0 = [min(r0max[i], center[i]) for i in range(len(r0max))]
    r1 = [min(r1max[i], volume_shape[i] - center[i]) for i in range(len(r0max))]
    patch_center = r0max

    if(len(center) == 3):
        output_volume[np.ix_(range(center[0] - r0[0], center[0] + r1[0]),
                             range(center[1] - r0[1], center[1] + r1[1]),
                             range(center[2] - r0[2], center[2] + r1[2]))] = \
            sub_volume[np.ix_(range(patch_center[0] - r0[0], patch_center[0] + r1[0]),
                              range(patch_center[1] - r0[1], patch_center[1] + r1[1]),
                              range(patch_center[2] - r0[2], patch_center[2] + r1[2]))]
    elif(len(center) == 4):
        output_volume[np.ix_(range(center[0] - r0[0], center[0] + r1[0]),
                             range(center[1] - r0[1], center[1] + r1[1]),
                             range(center[2] - r0[2], center[2] + r1[2]),
                             range(center[3] - r0[3], center[3] + r1[3]))] = \
            sub_volume[np.ix_(range(patch_center[0] - r0[0], patch_center[0] + r1[0]),
                              range(patch_center[1] - r0[1], patch_center[1] + r1[1]),
                              range(patch_center[2] - r0[2], patch_center[2] + r1[2]),
                              range(patch_center[3] - r0[3], patch_center[3] + r1[3]))]
    else:
        raise ValueError("array dimension should be 3 or 4")        
    return output_volume  

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

def load_nifty_volume_as_array(filename, with_header = False):
    """
    load nifty image into numpy array, and transpose it based on the [z,y,x] axis order
    The output array shape is like [Depth, Height, Width]
    inputs:
        filename: the input file name, should be *.nii or *.nii.gz
        with_header: return affine and hearder infomation
    outputs:
        data: a numpy data array
    """
    img = nibabel.load(filename)
    data = img.get_data()
    data = np.transpose(data, [2,1,0])
    if(with_header):
        return data, img.affine, img.header
    else:
        return data

def get_none_zero_region(im, margin):
    """
    get the bounding box of the non-zero region of an ND volume
    """
    input_shape = im.shape
    if(type(margin) is int ):
        margin = [margin]*len(input_shape)

    
    assert(len(input_shape) == len(margin))
    indxes = np.nonzero(im)


    idx_min = []
    idx_max = []
    for i in range(len(input_shape)):
        idx_min.append(indxes[i].min())
        idx_max.append(indxes[i].max())

    for i in range(len(input_shape)):
        idx_min[i] = max(idx_min[i] - margin[i], 0)
        idx_max[i] = min(idx_max[i] + margin[i], input_shape[i] - 1)
    return idx_min, idx_max

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

def crop_ND_volume_with_bounding_box(volume, min_idx, max_idx):
    """
    crop/extract a subregion form an nd image.
    """
    dim = len(volume.shape)
    assert(dim >= 2 and dim <= 5)
    if(dim == 2):
        output = volume[np.ix_(range(min_idx[0], max_idx[0] + 1),
                               range(min_idx[1], max_idx[1] + 1))]
    elif(dim == 3):
        output = volume[np.ix_(range(min_idx[0], max_idx[0] + 1),
                               range(min_idx[1], max_idx[1] + 1),
                               range(min_idx[2], max_idx[2] + 1))]
    elif(dim == 4):
        output = volume[np.ix_(range(min_idx[0], max_idx[0] + 1),
                               range(min_idx[1], max_idx[1] + 1),
                               range(min_idx[2], max_idx[2] + 1),
                               range(min_idx[3], max_idx[3] + 1))]
    elif(dim == 5):
        output = volume[np.ix_(range(min_idx[0], max_idx[0] + 1),
                               range(min_idx[1], max_idx[1] + 1),
                               range(min_idx[2], max_idx[2] + 1),
                               range(min_idx[3], max_idx[3] + 1),
                               range(min_idx[4], max_idx[4] + 1))]
    else:
        raise ValueError("the dimension number shoud be 2 to 5")
    return output

def get_random_roi_sampling_center(input_shape, output_shape, sample_mode='full', bounding_box = None):


    """
    get a random coordinate representing the center of a roi for sampling
    inputs:
        input_shape: the shape of sampled volume
        output_shape: the desired roi shape
        sample_mode: 'valid': the entire roi should be inside the input volume
                     'full': only the roi centre should be inside the input volume
        bounding_box: the bounding box which the roi center should be limited to
    outputs:
        center: the output center coordinate of a roi
    """
    center = []
    for i in range(len(input_shape)):


        if(sample_mode[i] == 'full'):
            if(bounding_box):
                x0 = bounding_box[i*2]; x1 = bounding_box[i*2 + 1]
            else:
                x0 = 0; x1 = input_shape[i]
        else:
            if(bounding_box):
                x0 = bounding_box[i*2] + int(output_shape[i]/2)   
                x1 = bounding_box[i*2+1] - int(output_shape[i]/2)   
            else:
                x0 = int(output_shape[i]/2)   
                x1 = input_shape[i] - x0
        if(x1 <= x0):
            centeri = int((x0 + x1)/2)
        else:
            centeri = random.randint(x0, x1)
        center.append(centeri)
    return center

def extract_roi_from_volume(volume, in_center, output_shape, fill = 'random'):
    """
    extract a roi from a 3d volume
    inputs:
        volume: the input 3D volume
        in_center: the center of the roi
        output_shape: the size of the roi
        fill: 'random' or 'zero', the mode to fill roi region where is outside of the input volume
    outputs:
        output: the roi volume
    """
    input_shape = volume.shape   
    if(fill == 'random'):
        output = np.random.normal(0, 1, size = output_shape)
    else:
        output = np.zeros(output_shape)
    r0max = [int(x/2) for x in output_shape]

    r1max = [output_shape[i] - r0max[i] for i in range(len(r0max))]
    r0 = [min(r0max[i], in_center[i]) for i in range(len(r0max))]
    r1 = [min(r1max[i], input_shape[i] - in_center[i]) for i in range(len(r0max))]
    out_center = r0max
    
    output[np.ix_(range(out_center[0] - r0[0], out_center[0] + r1[0]),
                  range(out_center[1] - r0[1], out_center[1] + r1[1]),
                  range(out_center[2] - r0[2], out_center[2] + r1[2]))] = \
        volume[np.ix_(range(in_center[0] - r0[0], in_center[0] + r1[0]),
                      range(in_center[1] - r0[1], in_center[1] + r1[1]),
                      range(in_center[2] - r0[2], in_center[2] + r1[2]))]
    return output

def save_to_pkl(probs, filename, outdir=""):
    """
    probs => [155, 240, 240]
    """
    if not os.path.exists("./{}".format(outdir)):
            os.mkdir("./{}".format(outdir))

    with open("./{}/{}.pkl".format(outdir, filename), 'wb') as f:
        pickle.dump(probs, f, protocol=pickle.HIGHEST_PROTOCOL)

def save_to_nii(im, filename, outdir="", mode="image", system="sitk"):
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


def sample_data(volume_list, label, weight):
    """
    sample 3d volume to size (depth, h, w, 4)
    """

    margin = 5
    volume_list = transpose_volumes(volume_list, DIRECTION)
    data_shape = PATCH_SIZE
    label_shape = PATCH_SIZE
    data_slice_number = data_shape[0]
    label_slice_number = label_shape[0]
    volume_shape = volume_list[0].shape
    sub_data_shape = [data_slice_number, data_shape[1], data_shape[2]]
    sub_label_shape =[label_slice_number, label_shape[1], label_shape[2]]

    label = transpose_volumes(label, DIRECTION)
    center_point = get_random_roi_sampling_center(volume_shape, sub_label_shape, "full", None)

    sub_label = extract_roi_from_volume(label,
                                        center_point,
                                        sub_label_shape,
                                            fill = 'zero')
    sub_data = []
    flip = False
    for moda in range(len(volume_list)):
        sub_data_moda = extract_roi_from_volume(volume_list[moda],center_point,sub_data_shape)
        if(flip):
            sub_data_moda = np.flip(sub_data_moda, -1)
        sub_data.append(sub_data_moda)
    sub_data = np.array(sub_data) #4, depth, h, w

    weight = transpose_volumes(weight, DIRECTION)
    sub_weight = extract_roi_from_volume(weight,
                                            center_point,
                                            sub_label_shape,
                                            fill = 'zero')
    if(flip):
        sub_weight = np.flip(sub_weight, -1)
    
    if(flip):
        sub_label = np.flip(sub_label, -1)
    batch = {}
    axis = [0,1,2,3] #[1,2,3,0] [d, h, w, modalities]
    batch['images']  = np.transpose(sub_data, axis)
    batch['weights'] = np.transpose(sub_weight[np.newaxis, ...], axis)
    batch['labels']  = np.transpose(sub_label[np.newaxis, ...], axis)
    # other meta info ?
    return [batch['images'], batch['weights'], batch['labels']]

def get_data_train(path_image, path_gt):
    volume_list, label, weight, _, _ = crop_brain_region(path_image, path_gt)
    images, weights, labels = sample_data(volume_list, label, weight)
    
    return images, weights, labels




def convert_block_to_point(path_save_point,volume_block,label_block,weight_block,x_global,y_global,z_global):
    x_axis, y_axis, z_axis,channel = volume_block.shape

    data_list = [[x+x_global,y+y_global,z+z_global,volume_block[x][y][z][0],volume_block[x][y][z][0],volume_block[x][y][z][0],volume_block[x][y][z][0],label_block[x][y][z]] for x in range(x_axis) for y in range(y_axis) for z in range(z_axis) if (weight_block[x][y][z]!=0)]
    
    current_number_point = len(data_list)
    pc_label = np.array(data_list)

    k_block = int(number_of_point/current_number_point) -1
    k_add = number_of_point - ((k_block+1)*current_number_point)
    dt_add = pc_label[0:k_add ,:]
    final_pc_label = np.concatenate((pc_label,dt_add))


    for i in range(k_block):
        final_pc_label = np.concatenate((final_pc_label,pc_label))

    xyz = final_pc_label[:, :3].astype(np.float32)
    colors = final_pc_label[:, 3:7].astype(np.uint8)
    labels = final_pc_label[:,7].astype(np.uint8)

    




    # #write full ply
    write_ply(path_save_point, (xyz, colors, labels), ['x', 'y', 'z', 't1ce', 't1', 'flair', 't2' ,'class'])

def save_list_to_txt(values,path_txt):
    with open(path_txt, 'w') as output:
        for row in values:
            output.write(str(row) + '\n')
list_name_IDs = []

total_block_64 = 0
total_block_64_None = 0
total_block_64_tumor = 0
def sample_volume_pointcloud(ID):

    # global total_block_64
    # global total_block_64_None
    # global total_block_64_tumor

    path_image = {'t1ce': '', 't1': '', 'flair': '', 't2': ''} 
    path_image['t1ce']=os.path.join(path_origin,ID,str(ID)+"_t1ce.nii.gz")
    path_image['t1']=os.path.join(path_origin,ID,str(ID)+"_t1.nii.gz")
    path_image['t2']=os.path.join(path_origin,ID,str(ID)+"_t2.nii.gz")
    path_image['flair']=os.path.join(path_origin,ID,str(ID)+"_flair.nii.gz")
    path_gt = os.path.join(path_origin,ID,str(ID)+"_seg.nii.gz")
    volume, label, weight, _, _ = crop_brain_region(path_image, path_gt)
    volume = np.moveaxis(np.array(volume), 0, 3)


    # for i in range(weight.shape[0]):
    #     # weight_id = weight[i]
    #     plt.imsave("/home/ubuntu/Brain_Point/RandLA-Net/utils/img_2D/mask_0"+str(i)+".png", weight[i])

    # print("label ",np.unique(label))
    # exit()
    

    x_axis,y_axis,z_axis = volume.shape[0],volume.shape[1],volume.shape[2]
    count = 0
    conut_label = 0
    stride = 54
    stride_tumor = 4
    for x in range(0,x_axis,stride):
        if x+64>x_axis:
            x = x_axis-64
        for y in range(0,y_axis,stride):
            if y+64>y_axis:
                y = y_axis-64
            for z in range(0,z_axis, stride):
                if z+64>z_axis:
                    z = z_axis-64
                volume_block = volume[x:x+64,y:y+64,z:z+64,:].copy()
                label_block = label[x:x+64,y:y+64,z:z+64].copy()
                weight_block = weight[x:x+64,y:y+64,z:z+64].copy()
                if max(np.unique(weight_block))==0:
                    continue
                name_ID = str(ID)+"_xyz_"+str(x)+"_"+str(y)+"_"+str(z)+".ply"
                path_save_point = os.path.join(output_savepoint,str(ID)+"_xyz_"+str(x)+"_"+str(y)+"_"+str(z)+".ply")

                # convert_block_to_point(path_save_point,volume_block,label_block,weight_block,x,y,z)
                convert_block_to_point(path_save_point,volume_block,label_block,weight_block,0,0,0)
                # if max(np.unique(label_block)>0):
                if  np.count_nonzero(label_block > 0)>0:
                    conut_label +=1

                if np.count_nonzero(label_block > 0) >= ((64*64*64)/20):
                    
                    # print(conut_label, "tumor ", np.count_nonzero(label_block > 0))

                    
                    stride = stride_tumor
                else:
                    stride = 54
                #     print("np.unique(label_temp) ",np.unique(label_temp))
                # print("name_ID ",name_ID)
                count +=1
                list_name_IDs.append(name_ID)
    print(ID,"tumor : ", conut_label , "none tumor ", count-conut_label)
    # total_block_64 = total_block_64 + count
    # total_block_64_None = (count-conut_label) + conut_label
    # total_block_64_tumor = total_block_64_tumor + conut_label




#     print("images.shape ",volume.shape)
#     print("weights.shape ",label.shape)
#     print("labels.shape ",weight.shape)    


# sample_volume_pointcloud("BraTS20_Training_313")


# list_ID = os.listdir(path_origin)
# train_IDs, val_IDs = train_test_split(list_ID,train_size=0.8,test_size=0.2, random_state=182)

# save_IDs_point_cloud = "/home/ubuntu/Research/BraTS/3D_Medical_Segmentation/utils/data_splited/train_IDs_point_cloud.txt"

path_train_ID = "/home/ubuntu/Research/BraTS/3D_Medical_Segmentation/utils/data_splited/val_IDs_fold1_82_182.txt"
with open(path_train_ID) as f:
    content = f.readlines()
train_IDs = [x.strip() for x in content] 
# train_IDs = train_IDs[:2]




if __name__ == '__main__':

    # Run with one core
    parallel = True
    if parallel:
        # with concurrent.futures.ProcessPoolExecutor(20) as executor:
        #     tqdm(executor.map(sample_volume_pointcloud, train_IDs), total=len(train_IDs))

        with concurrent.futures.ProcessPoolExecutor(20) as executor:
            tqdm(executor.map(sample_volume_pointcloud, train_IDs), total=len(train_IDs))

    else:
        for i,ID in enumerate(train_IDs):
            print("processing ",i," / ",len(train_IDs))
            # conver_point_vs(load_volume(ID),ID)
            # exit()
            sample_volume_pointcloud(ID)
    # save_list_to_txt(list_name_IDs,save_IDs_point_cloud)
    # print("total_block_64 : ",total_block_64)
    # print("total_block_64_None : ",total_block_64_None )
    # print("total_block_64_tumor : ",total_block_64_tumor)

# data = {}
# data['image_data'] =  {'t1ce': '', 't1': '', 'flair': '', 't2': ''} 
# data['gt'] =  ''

# data['image_data']['t1ce'] = "heloo"

# print(data['image_data'])