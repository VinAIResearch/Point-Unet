from keras.models import *
from attention import *
from bilinear_upsampling import BilinearUpsampling, BilinearUpsampling3D
import tensorflow as tf
from tensorpack.tfutils.scope_utils import auto_reuse_variable_scope
from tensorpack.tfutils.summary import add_moving_summary
from tensorpack.tfutils.argscope import argscope
from tensorpack.tfutils.scope_utils import under_name_scope

from tensorpack.models import (
    layer_register
)
from custom_ops import BatchNorm3d, InstanceNorm5d
import numpy as np
import config
import tensorflow.contrib.slim as slim
PADDING = "SAME"
DATA_FORMAT="channels_last"
BASE_FILTER = 16

@layer_register(log_shape=True)
def unet3d(inputs):
    print("inputs ", inputs)
    depth = config.DEPTH
    filters = []
    down_list = []
    deep_supervision = None
    layer = tf.layers.conv3d(inputs=inputs, 
                   filters=BASE_FILTER,
                   kernel_size=(3,3,3),
                   strides=1,
                   padding=PADDING,
                   activation=lambda x, name=None: BN_Relu(x),
                   data_format=DATA_FORMAT,
                   name="init_conv")
    print(layer.name, layer.shape[1:])
    for d in range(depth):
        if config.FILTER_GROW:
            num_filters = BASE_FILTER * (2**d)
        else:
            num_filters = BASE_FILTER
        filters.append(num_filters)
        layer = Unet3dBlock('down{}'.format(d), layer, kernels=(3,3,3), n_feat=num_filters, s=1)
        print("Down_Unet ",d,"    ",layer.shape[1:])
        down_list.append(layer)
        if d != depth - 1:
            layer = tf.layers.conv3d(inputs=layer, 
                                    filters=num_filters*2,
                                    kernel_size=(3,3,3),
                                    strides=(2,2,2),
                                    padding=PADDING,
                                    activation=lambda x, name=None: BN_Relu(x),
                                    data_format=DATA_FORMAT,
                                    name="stride2conv{}".format(d))
        print("Down Conv3D ",d, "   ", layer.shape[1:])
        # print(layer.name,layer.shape[1:])
    for d in range(depth-2, -1, -1):
        layer = UnetUpsample(d, layer, filters[d])
        print("Up_Unet ",d,"    ",layer.shape[1:])
        if DATA_FORMAT == 'channels_first':
            layer = tf.concat([layer, down_list[d]], axis=1)
        else:
            layer = tf.concat([layer, down_list[d]], axis=-1)
            
        print("concat ",d, down_list[d].shape ," TO ", layer.shape[1:])
        #layer = Unet3dBlock('up{}'.format(d), layer, kernels=(3,3,3), n_feat=filters[d], s=1)
        layer = tf.layers.conv3d(inputs=layer, 
                                filters=filters[d],
                                kernel_size=(3,3,3),
                                strides=1,
                                padding=PADDING,
                                activation=lambda x, name=None: BN_Relu(x),
                                data_format=DATA_FORMAT,
                                name="lo_conv0_{}".format(d))
        print("Up1 Conv3d ",d, layer.shape[1:])
        layer = tf.layers.conv3d(inputs=layer, 
                                filters=filters[d],
                                kernel_size=(1,1,1),
                                strides=1,
                                padding=PADDING,
                                activation=lambda x, name=None: BN_Relu(x),
                                data_format=DATA_FORMAT,
                                name="lo_conv1_{}".format(d))
        print("Up2 Conv3d ",d, layer.shape[1:])

        if config.DEEP_SUPERVISION:
            if d < 3 and d > 0:
                pred = tf.layers.conv3d(inputs=layer, 
                                    filters=config.NUM_CLASS,
                                    kernel_size=(1,1,1),
                                    strides=1,
                                    padding=PADDING,
                                    activation=tf.identity,
                                    data_format=DATA_FORMAT,
                                    name="deep_super_{}".format(d))
                print("pred  ",d, pred.shape[1:])
                # print("deep_supervision before ",d, deep_supervision.shape[1:])
                if deep_supervision is None:
                    deep_supervision = pred
                else:
                    deep_supervision = deep_supervision + pred
                deep_supervision = Upsample3D(d, deep_supervision)
                # print("deep_supervision  after ",d, deep_supervision.shape[1:])
                
    layer = tf.layers.conv3d(layer, 
                            filters=config.NUM_CLASS,
                            kernel_size=(1,1,1),
                            padding="SAME",
                            activation=tf.identity,
                            data_format=DATA_FORMAT,
                            name="final")
    print("layer_er ", layer.shape[1:])
    if config.DEEP_SUPERVISION:
        layer = layer + deep_supervision
    if DATA_FORMAT == 'channels_first':
        layer = tf.transpose(layer, [0, 2, 3, 4, 1]) # to-channel last
    print("final", layer.shape[1:]) # [3, num_class, d, h, w]

    return layer

class Copy(Layer):
    def call(self, inputs, **kwargs):
        copy = tf.identity(inputs)
        return copy
    def compute_output_shape(self, input_shape):
        return input_shape

class layertile(Layer):
    def call(self, inputs, **kwargs):
        image = tf.reduce_mean(inputs, axis=-1)
        image = tf.expand_dims(image, -1)
        image = tf.tile(image, [1, 1, 1, 32])
        return image

    def compute_output_shape(self, input_shape):
        output_shape = list(input_shape)[:-1] + [32]
        return tuple(output_shape)

def AtrousBlock3D(input_tensor, filters, rate, block_id, stride=1):
    x = tf.layers.conv3d(inputs=input_tensor, 
                   filters=filters,
                   kernel_size=(3,3,3),
                   strides=(stride, stride, stride),
                   dilation_rate=(rate, rate, rate),
                   padding=PADDING,
                   use_bias=False,
                   activation=lambda x, name=None: BN_Relu(x),
                   data_format=DATA_FORMAT,
                   name=block_id + "_dilation")
    # x = Conv3D(filters, (3, 3, 3), strides=(stride, stride, stride), dilation_rate=(rate, rate, rate),
    #            padding='same', use_bias=False, name=block_id + '_dilation')(input_tensor)
    return x

def CFE3D(input_tensor, filters, block_id):
    rate = [3, 5, 7]
    cfe0 = tf.layers.conv3d(inputs=input_tensor, 
                   filters=filters,
                   kernel_size=(1,1,1),
                   use_bias=False,
                   strides=1,
                   padding=PADDING,
                   activation=lambda x, name=None: BN_Relu(x),
                   data_format=DATA_FORMAT,
                   name=block_id + "_cfe0")
    # cfe0 = Conv3D(filters, (1, 1, 1), padding='same', use_bias=False, name=block_id + '_cfe0')(
    #     input_tensor)
    cfe1 = AtrousBlock3D(input_tensor, filters, rate[0], block_id + '_cfe1')
    cfe2 = AtrousBlock3D(input_tensor, filters, rate[1], block_id + '_cfe2')
    cfe3 = AtrousBlock3D(input_tensor, filters, rate[2], block_id + '_cfe3')
    cfe_concat = tf.concat([cfe0, cfe1, cfe2, cfe3], axis=-1, name=block_id + 'concatcfe')
    #cfe_concat = Concatenate(name=block_id + 'concatcfe', axis=-1)([cfe0, cfe1, cfe2, cfe3])
    # with tf.variable_scope(block_id + "_BN") as scope:
    #     cfe_concat = BN_Relu(cfe_concat)
    return cfe_concat

@layer_register(log_shape=True)
def unet3d_attention(inputs):
    print("inputs ", inputs)
    depth = config.DEPTH
    filters = []
    down_list = []
    layer = tf.layers.conv3d(inputs=inputs, 
                   filters=BASE_FILTER,
                   kernel_size=(3,3,3),
                   strides=1,
                   padding=PADDING,
                   activation=lambda x, name=None: BN_Relu(x),
                   data_format=DATA_FORMAT,
                   name="init_conv")
    print(layer.name, layer.shape[1:])
    for d in range(depth):
        if config.FILTER_GROW:
            num_filters = BASE_FILTER * (2**d)
        else:
            num_filters = BASE_FILTER
        filters.append(num_filters)
        layer = Unet3dBlock('down{}'.format(d), layer, kernels=(3,3,3), n_feat=num_filters, s=1)
        print("Down_Unet ",d,"    ",layer.shape[1:])
        down_list.append(layer)
        if d != depth - 1:
            layer = tf.layers.conv3d(inputs=layer, 
                                    filters=num_filters*2,
                                    kernel_size=(3,3,3),
                                    strides=(2,2,2),
                                    padding=PADDING,
                                    activation=lambda x, name=None: BN_Relu(x),
                                    data_format=DATA_FORMAT,
                                    name="stride2conv{}".format(d))
            print("Down Conv3D ",d, "   ", layer.shape[1:])
        # print(layer.name,layer.shape[1:])

    C1 = tf.layers.conv3d(inputs=down_list[0], 
                   filters=64,
                   kernel_size=(3,3,3),
                   strides=1,
                   padding=PADDING,
                   activation=lambda x, name=None: BN_Relu(x),
                   data_format=DATA_FORMAT,
                   name="C1_conv")

    C2 = tf.layers.conv3d(inputs=down_list[1], 
                   filters=64,
                   kernel_size=(3,3,3),
                   strides=1,
                   padding=PADDING,
                   activation=lambda x, name=None: BN_Relu(x),
                   data_format=DATA_FORMAT,
                   name="C2_conv")

    #C1 = Conv3D(64, (3, 3, 3), padding='same', name='C1_conv')(down_list[0])
    print("Low level feature 1\t", C1.shape[1:])
    # with tf.variable_scope("C1_BN") as scope:
    #     C1 = BN_Relu(C1)
    #C2 = Conv3D(64, (3, 3, 3), padding='same', name='C2_conv')(down_list[1])
    print("Low level feature 2\t", C2.shape[1:])
    # with tf.variable_scope("C2_BN") as scope:
    #     C2 = BN_Relu(C2)

    C3_cfe = CFE3D(down_list[2], 32, 'C3_cfe')
    print("High level feature 1 CFE\t", C3_cfe.shape[1:])
    C4_cfe = CFE3D(down_list[3], 32, 'C4_cfe')
    print("High level feature 2 CFE\t", C4_cfe.shape[1:])
    C5_cfe = CFE3D(down_list[4], 32, 'C5_cfe')
    print("High level feature 3 CFE\t", C5_cfe.shape[1:])
    # C5_cfe = BilinearUpsampling3D(upsampling=(4, 4, 4), name='C5_cfe_up4')(C5_cfe)
    # C4_cfe = BilinearUpsampling3D(upsampling=(2, 2, 2), name='C4_cfe_up2')(C4_cfe)
    C5_cfe = UnetUpsample('C5_cfe_up4', C5_cfe, 4, 128)
    C4_cfe = UnetUpsample('C4_cfe_up2', C4_cfe, 2, 128)
    # C5_cfe = BilinearUpsampling3D(C5_cfe, 4, name='C5_cfe_up4')
    # C4_cfe = BilinearUpsampling3D(C4_cfe, 2, name='C4_cfe_up2')
    C345 = tf.concat([C3_cfe, C4_cfe, C5_cfe], axis=-1, name='C345_aspp_concat')
    #C345 = Concatenate(name='C345_aspp_concat', axis=-1)([C3_cfe, C4_cfe, C5_cfe])
    print("High level features aspp concat\t", C345.shape[1:])

    if config.CA_attention:
        C345 = ChannelWiseAttention3D(C345, name='C345_ChannelWiseAttention_withcpfe')
        print('High level features CA\t', C345.shape[1:])

    C345 = tf.layers.conv3d(inputs=C345, 
                   filters=64,
                   kernel_size=(1,1,1),
                   strides=1,
                   padding=PADDING,
                   activation=lambda x, name=None: BN_Relu(x),
                   data_format=DATA_FORMAT,
                   name="C345_conv")
    #C345 = Conv3D(64, (1, 1, 1), padding='same', name='C345_conv')(C345)
    print('High level features conv\t', C345.shape[1:])
    # with tf.variable_scope("C345_BN") as scope:
    #     C345 = BN_Relu(C345)
    C345 = UnetUpsample('C345_up4', C345, 4, 64)
    #C345 = BilinearUpsampling3D(C345, 4, name='C345_up4')
    print('High level features upsampling\t', C345.shape[1:])

    if config.SA_attention:
        SA = SpatialAttention3D(C345, 'spatial_attention')
        print('High level features SA\t', SA.shape[1:])
    C2 = UnetUpsample('C2_up2', C2, 2, 64)
    #C2 = BilinearUpsampling3D(C2, 2, name='C2_up2')
    C12 = tf.concat([C1, C2], axis=-1, name='C12_concat')
    #C12 = Concatenate(name='C12_concat', axis=-1)([C1, C2])
    C12 = tf.layers.conv3d(inputs=C12, 
                   filters=64,
                   kernel_size=(3,3,3),
                   strides=1,
                   padding=PADDING,
                   activation=lambda x, name=None: BN_Relu(x),
                   data_format=DATA_FORMAT,
                   name="C12_conv")
    #C12 = Conv3D(64, (3, 3, 3), padding='same', name='C12_conv')(C12)
    print('Low level feature conv\t', C12.shape[1:])
    # with tf.variable_scope("C12_BN") as scope:
    #     C12 = BN_Relu(C12)
    if config.SA_attention:
        C12 = tf.math.multiply(SA, C12, name='C12_atten_mutiply')
    #C12 = Multiply(name='C12_atten_mutiply')([SA, C12])

    fea = tf.concat([C12, C345], axis=-1, name='fuse_concat')
    #fea = Concatenate(name='fuse_concat',axis=-1)([C12, C345])
    print('Low + High level feature\t', fea.shape[1:])
    layer = tf.layers.conv3d(fea, 
                            filters=config.NUM_CLASS,
                            kernel_size=(3,3,3),
                            padding="SAME",
                            activation=tf.identity,
                            data_format=DATA_FORMAT,
                            name="final")
    #layer = Conv3D(config.NUM_CLASS, (3, 3, 3), padding='same', name='sa')(fea)

    if DATA_FORMAT == 'channels_first':
        layer = tf.transpose(layer, [0, 2, 3, 4, 1]) # to-channel last
    print("final", layer.shape[1:]) # [3, num_class, d, h, w]

    return layer

def Upsample3D(prefix, l, scale=2):
    l = tf.keras.layers.UpSampling3D(size=(scale,scale,scale), data_format=DATA_FORMAT)(l)
    """
    l = tf.layers.conv3d_transpose(inputs=l, 
                                filters=config.NUM_CLASS,
                                kernel_size=(2,2,2),
                                strides=2,
                                padding=PADDING,
                                activation=tf.nn.relu,
                                data_format=DATA_FORMAT,
                                name="upsampe_{}".format(prefix))
    
    l_out = tf.identity(l)
    if DATA_FORMAT == 'channels_first':
        l = tf.transpose(l, [0, 2, 3, 4, 1])
    l_shape = l.get_shape().as_list()
    l = tf.reshape(l, [l_shape[0]*l_shape[1], l_shape[2], l_shape[3], l_shape[4]])
    l = tf.image.resize_images(l , (l_shape[2]*scale, l_shape[3]*scale))
    l = tf.reshape(l, [l_shape[0], l_shape[1], l_shape[2]*scale, l_shape[3]*scale, l_shape[4]])
    if DATA_FORMAT == 'channels_first':
        l = tf.transpose(l, [0, 4, 1, 2, 3]) # Back to channel_first
    """
    return l

def UnetUpsample(prefix, l, scale, num_filters):
#def UnetUpsample(prefix, l, num_filters):
    """
    l = tf.layers.conv3d_transpose(inputs=l, 
                                filters=num_filters,
                                kernel_size=(2,2,2),
                                strides=2,
                                padding=PADDING,
                                activation=tf.nn.relu,
                                data_format=DATA_FORMAT,
                                name="up_conv0_{}".format(prefix))
    """
    # print(" l-1 ",prefix, l )
    l = Upsample3D('', l, scale)
    #l = Upsample3D('', l)
    # print(" l-2 ",prefix, l )
    l = tf.layers.conv3d(inputs=l, 
                        filters=num_filters,
                        kernel_size=(3,3,3),
                        strides=1,
                        padding=PADDING,
                        activation=lambda x, name=None: BN_Relu(x),
                        data_format=DATA_FORMAT,
                        name="up_conv1_{}".format(prefix))
    return l

def BN_Relu(x):
    if config.INSTANCE_NORM:
        l = InstanceNorm5d('ins_norm', x, data_format=DATA_FORMAT)
    else:
        l = BatchNorm3d('bn', x, axis=1 if DATA_FORMAT == 'channels_first' else -1)
    l = tf.nn.relu(l)
    return l

def Unet3dBlock(prefix, l, kernels, n_feat, s):
    if config.RESIDUAL:
        l_in = l

    for i in range(2):
        l = tf.layers.conv3d(inputs=l, 
                   filters=n_feat,
                   kernel_size=kernels,
                   strides=1,
                   padding=PADDING,
                   activation=lambda x, name=None: BN_Relu(x),
                   data_format=DATA_FORMAT,
                   name="{}_conv_{}".format(prefix, i))

    return l_in + l if config.RESIDUAL else l

### from niftynet ####
def labels_to_one_hot(ground_truth, num_classes=1):
    """
    Converts ground truth labels to one-hot, sparse tensors.
    Used extensively in segmentation losses.
    :param ground_truth: ground truth categorical labels (rank `N`)
    :param num_classes: A scalar defining the depth of the one hot dimension
        (see `depth` of `tf.one_hot`)
    :return: one-hot sparse tf tensor
        (rank `N+1`; new axis appended at the end)
    """
    # read input/output shapes
    if isinstance(num_classes, tf.Tensor):
        num_classes_tf = tf.to_int32(num_classes)
    else:
        num_classes_tf = tf.constant(num_classes, tf.int32)
    input_shape = tf.shape(ground_truth)
    output_shape = tf.concat(
        [input_shape, tf.reshape(num_classes_tf, (1,))], 0)

    if num_classes == 1:
        # need a sparse representation?
        return tf.reshape(ground_truth, output_shape)

    # squeeze the spatial shape
    ground_truth = tf.reshape(ground_truth, (-1,))
    # shape of squeezed output
    dense_shape = tf.stack([tf.shape(ground_truth)[0], num_classes_tf], 0)

    # create a rank-2 sparse tensor
    ground_truth = tf.to_int64(ground_truth)
    ids = tf.range(tf.to_int64(dense_shape[0]), dtype=tf.int64)
    ids = tf.stack([ids, ground_truth], axis=1)
    one_hot = tf.SparseTensor(
        indices=ids,
        values=tf.ones_like(ground_truth, dtype=tf.float32),
        dense_shape=tf.to_int64(dense_shape))

    # resume the spatial dims
    one_hot = tf.sparse_reshape(one_hot, output_shape)
    return one_hot

def generalised_dice_loss(prediction,
                          ground_truth,
                          weight_map=None,
                          type_weight='Square'):

    """
    Function to calculate the Generalised Dice Loss defined in
        Sudre, C. et. al. (2017) Generalised Dice overlap as a deep learning
        loss function for highly unbalanced segmentations. DLMIA 2017
    :param prediction: the logits
    :param ground_truth: the segmentation ground truth
    :param weight_map:
    :param type_weight: type of weighting allowed between labels (choice
        between Square (square of inverse of volume),
        Simple (inverse of volume) and Uniform (no weighting))
    :return: the loss
    """
    prediction = tf.cast(prediction, tf.float32)
    if len(ground_truth.shape) == len(prediction.shape):
        ground_truth = ground_truth[..., -1]
    one_hot = labels_to_one_hot(ground_truth, tf.shape(prediction)[-1])

    if weight_map is not None:
        n_classes = prediction.shape[1].value
        weight_map_nclasses = tf.reshape(
            tf.tile(weight_map, [n_classes]), prediction.get_shape())
        ref_vol = tf.sparse_reduce_sum(
            weight_map_nclasses * one_hot, reduction_axes=[0])

        intersect = tf.sparse_reduce_sum(
            weight_map_nclasses * one_hot * prediction, reduction_axes=[0])
        seg_vol = tf.reduce_sum(
            tf.multiply(weight_map_nclasses, prediction), 0)
    else:
        ref_vol = tf.sparse_reduce_sum(one_hot, reduction_axes=[0])
        intersect = tf.sparse_reduce_sum(one_hot * prediction,
                                         reduction_axes=[0])
        seg_vol = tf.reduce_sum(prediction, 0)
    if type_weight == 'Square':
        weights = tf.reciprocal(tf.square(ref_vol))
    elif type_weight == 'Simple':
        weights = tf.reciprocal(ref_vol)
    elif type_weight == 'Uniform':
        weights = tf.ones_like(ref_vol)
    else:
        raise ValueError("The variable type_weight \"{}\""
                         "is not defined.".format(type_weight))
    new_weights = tf.where(tf.is_inf(weights), tf.zeros_like(weights), weights)
    weights = tf.where(tf.is_inf(weights), tf.ones_like(weights) *
                       tf.reduce_max(new_weights), weights)
    generalised_dice_numerator = \
        2 * tf.reduce_sum(tf.multiply(weights, intersect))
    generalised_dice_denominator = \
        tf.reduce_sum(tf.multiply(weights, seg_vol + ref_vol)) + 1e-6
    generalised_dice_score = \
        generalised_dice_numerator / generalised_dice_denominator
    return 1 - generalised_dice_score


def dice(prediction, ground_truth, weight_map=None):


    # with tf.Session() as sess:  
    #     print(ground_truth.eval()) 
        # print(prediction.eval()) 
    """
    Function to calculate the dice loss with the definition given in
        Milletari, F., Navab, N., & Ahmadi, S. A. (2016)
        V-net: Fully convolutional neural
        networks for volumetric medical image segmentation. 3DV 2016
    using a square in the denominator
    :param prediction: the logits
    :param ground_truth: the segmentation ground_truth
    :param weight_map:
    :return: the loss
    """



    ground_truth = tf.to_int64(ground_truth)
    prediction = tf.cast(prediction, tf.float32)
    
    ids = tf.range(tf.to_int64(tf.shape(ground_truth)[0]), dtype=tf.int64)

    # print(prediction.eval()) 
    ids = tf.stack([ids, ground_truth], axis=1)

    # with tf.Session() as sess:  
    #     print(ids.eval()) 
    one_hot = tf.SparseTensor(
        indices=ids,
        values=tf.ones_like(ground_truth, dtype=tf.float32),
        dense_shape=tf.to_int64(tf.shape(prediction)))


    if weight_map is not None:
        n_classes = prediction.shape[1].value
        weight_map_nclasses = tf.reshape(
            tf.tile(weight_map, [n_classes]), prediction.get_shape())
        dice_numerator = 2.0 * tf.sparse_reduce_sum(
            weight_map_nclasses * one_hot * prediction, reduction_axes=[0])
        dice_denominator = \
            tf.reduce_sum(weight_map_nclasses * tf.square(prediction),
                          reduction_indices=[0]) + \
            tf.sparse_reduce_sum(one_hot * weight_map_nclasses,
                                 reduction_axes=[0])
    else:
        dice_numerator = 2.0 * tf.sparse_reduce_sum(
            one_hot * prediction, reduction_axes=[0])
        dice_denominator = \
            tf.reduce_sum(tf.square(prediction), reduction_indices=[0]) + \
            tf.sparse_reduce_sum(one_hot, reduction_axes=[0])
    epsilon_denominator = 0.00001

    dice_score = dice_numerator / (dice_denominator + epsilon_denominator)
    
    return 1.0 - tf.reduce_mean(dice_score)

def dice_mixup(prediction, ground_truth, weight_map=None):


    # with tf.Session() as sess:  
    #     print(ground_truth.eval()) 
        # print(prediction.eval()) 
    """
    Function to calculate the dice loss with the definition given in
        Milletari, F., Navab, N., & Ahmadi, S. A. (2016)
        V-net: Fully convolutional neural
        networks for volumetric medical image segmentation. 3DV 2016
    using a square in the denominator
    :param prediction: the logits
    :param ground_truth: the segmentation ground_truth
    :param weight_map:
    :return: the loss
    """
    prediction = tf.cast(prediction, tf.float32)

    if weight_map is not None:
        n_classes = prediction.shape[1].value
        weight_map_nclasses = tf.reshape(
            tf.tile(weight_map, [n_classes]), prediction.get_shape())
        dice_numerator = 2.0 * tf.reduce_sum(
            weight_map_nclasses * ground_truth * prediction, axis=[0])
        dice_denominator = \
            tf.reduce_sum(weight_map_nclasses * tf.square(prediction),
                          reduction_indices=[0]) + \
            tf.reduce_sum(ground_truth * weight_map_nclasses,
                                 axis=[0])
    else:
        dice_numerator = 2.0 * tf.sparse_reduce_sum(
            one_hot * prediction, reduction_axes=[0])
        dice_denominator = \
            tf.reduce_sum(tf.square(prediction), reduction_indices=[0]) + \
            tf.sparse_reduce_sum(one_hot, reduction_axes=[0])
    epsilon_denominator = 0.00001

    dice_score = dice_numerator / (dice_denominator + epsilon_denominator)
    
    return 1.0 - tf.reduce_mean(dice_score)

def Loss(feature, weight, gt):


    # compute batch-wise
    losses = []
    for idx in range(config.BATCH_SIZE):
        f = tf.reshape(feature[idx], [-1, config.NUM_CLASS])
        #f = tf.cast(f, dtype=tf.float32)
        #f = tf.nn.softmax(f)
        w = tf.reshape(weight[idx], [-1])
        if config.MIXUP:
            g = tf.reshape(gt[idx], [-1, config.NUM_CLASS])
        else:
            g = tf.reshape(gt[idx], [-1])
        if g.shape.as_list()[-1] == 1:
            g = tf.squeeze(g, axis=-1) # (nvoxel, )
        if w.shape.as_list()[-1] == 1:
            w = tf.squeeze(w, axis=-1) # (nvoxel, )
        f = tf.nn.softmax(f)
        if config.MIXUP:
            loss_per_batch = dice_mixup(f, g, weight_map=w)
        else:
            loss_per_batch = dice(f, g, weight_map=w)
        # loss_per_batch = cross_entropy(f, g, weight_map=w)
        losses.append(loss_per_batch)
        
    return tf.reduce_mean(losses, name="dice_loss")

# def dice(prediction, ground_truth, weight_map=None):






#     # ground_truth = tf.to_int64(ground_truth)
#     prediction = tf.cast(prediction, tf.float32)
    
#     # ground_truth = tf.cast(ground_truth, tf.float32)
#     ground_truth = prediction
#     # ground_truth = prediction
    
#     # ids = tf.range(tf.to_int64(tf.shape(ground_truth)[0]), dtype=tf.int64)

#     # # print(prediction.eval()) 
#     # ids = tf.stack([ids, ground_truth], axis=1)

#     # with tf.Session() as sess:  
#     #     print(ids.eval()) 
#     # one_hot = tf.SparseTensor(
#     #     indices=ids,
#     #     values=tf.ones_like(ground_truth, dtype=tf.float32),
#     #     dense_shape=tf.to_int64(tf.shape(prediction)))


#     # if weight_map is not None:
#     #     n_classes = prediction.shape[1].value
#     #     weight_map_nclasses = tf.reshape(
#     #         tf.tile(weight_map, [n_classes]), prediction.get_shape())
#     #     dice_numerator = 2.0 * tf.sparse_reduce_sum(
#     #         weight_map_nclasses * one_hot * prediction, reduction_axes=[0])
#     #     dice_denominator = \
#     #         tf.reduce_sum(weight_map_nclasses * tf.square(prediction),
#     #                       reduction_indices=[0]) + \
#     #         tf.sparse_reduce_sum(one_hot * weight_map_nclasses,
#     #                              reduction_axes=[0])
#     # else:
#     dice_numerator = 2.0 * tf.reduce_sum(tf.square(prediction), reduction_indices=[0])
#     print("dice_numerator ",dice_numerator)
#     dice_denominator = \
#         tf.reduce_sum(tf.square(ground_truth), reduction_indices=[0]) + \
#         tf.reduce_sum(tf.square(prediction), reduction_indices=[0])
#     epsilon_denominator = 0.00001

#     dice_score = dice_numerator / (dice_denominator + epsilon_denominator)
    
#     return 1.0 - tf.reduce_mean(dice_score)

# def Loss(feature, weight, gt):




  
#     # compute batch-wise
#     losses = []
#     for idx in range(config.BATCH_SIZE):
#         f = tf.reshape(feature[idx], [-1, config.NUM_CLASS])
#         #f = tf.cast(f, dtype=tf.float32)
#         #f = tf.nn.softmax(f)
#         w = tf.reshape(weight[idx], [-1])
#         g = tf.reshape(gt[idx], [-1, config.NUM_CLASS])
#         # g = tf.reshape(gt[idx], [-1])
#         # if g.shape.as_list()[-1] == 1:
#         #     g = tf.squeeze(g, axis=-1) # (nvoxel, )
#         if w.shape.as_list()[-1] == 1:
#             w = tf.squeeze(w, axis=-1) # (nvoxel, )
#         f = tf.nn.softmax(f)
#         g = tf.nn.softmax(g)
#         loss_per_batch = dice(f, g, weight_map=w)
#         # loss_per_batch = cross_entropy(f, g, weight_map=w)
#         losses.append(loss_per_batch)
#     # print("feature : ", feature)
#     # print("gt : ", gt)
#     # exit()
#     return tf.reduce_mean(losses, name="dice_loss")

    
if __name__ == "__main__":
    #image = tf.transpose(tf.constant(np.zeros((config.BATCH_SIZE,160,208,176,4)).astype(np.float32)), [0,4,1,2,3])
    image = tf.constant(np.zeros((config.BATCH_SIZE,160,208,176,4)).astype(np.float32))
    #gt = tf.constant(np.zeros((config.BATCH_SIZE,160,208,176,4)).astype(np.float32))
    gt = tf.constant(np.zeros((config.BATCH_SIZE,160,208,176,1)).astype(np.float32))
    weight = tf.constant(np.ones((config.BATCH_SIZE,160,208,176,1)).astype(np.float32))
    t = unet3d_attention('unet3d_attention', image)
    loss = Loss(t, weight, gt)
    print("output size : ",t.shape[1:],"\n Loss : ", loss)