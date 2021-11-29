import tensorflow as tf
from keras.engine import Layer
from keras.layers import *
from bilinear_upsampling import BilinearUpsampling

import config
from custom_ops import BatchNorm3d, InstanceNorm5d

PADDING = "SAME"
DATA_FORMAT="channels_last"

class BatchNorm(BatchNormalization):
    def call(self, inputs, training=None):
          return super(self.__class__, self).call(inputs, training=True)

def BN(input_tensor,block_id):
    bn = BatchNorm(name=block_id+'_BN')(input_tensor)
    a = Activation('relu',name=block_id+'_relu')(bn)
    return a

def BN_Relu(x):
    if config.INSTANCE_NORM:
        l = InstanceNorm5d('ins_norm', x, data_format=DATA_FORMAT)
    else:
        l = BatchNorm3d('bn', x, axis=1 if DATA_FORMAT == 'channels_first' else -1)
    l = tf.nn.relu(l)
    return l

def l1_reg(weight_matrix):
    return K.mean(weight_matrix)

class Repeat(Layer):
    def __init__(self,repeat_list, **kwargs):
        super(Repeat, self).__init__(**kwargs)
        self.repeat_list = repeat_list

    def call(self, inputs):
        outputs = tf.tile(inputs, self.repeat_list)
        return outputs
    def get_config(self):
        config = {
            'repeat_list': self.repeat_list
        }
        base_config = super(Repeat, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

    def compute_output_shape(self, input_shape):
        output_shape = [None]
        for i in range(1,len(input_shape)):
            output_shape.append(input_shape[i]*self.repeat_list[i])
        return tuple(output_shape)

def SpatialAttention(inputs,name):
    k = 9
    H, W, C = map(int,inputs.get_shape()[1:])
    attention1 = Conv2D(C / 2, (1, k), padding='same', name=name+'_1_conv1')(inputs)
    attention1 = BN(attention1,'attention1_1')
    attention1 = Conv2D(1, (k, 1), padding='same', name=name + '_1_conv2')(attention1)
    attention1 = BN(attention1, 'attention1_2')
    attention2 = Conv2D(C / 2, (k, 1), padding='same', name=name + '_2_conv1')(inputs)
    attention2 = BN(attention2, 'attention2_1')
    attention2 = Conv2D(1, (1, k), padding='same', name=name + '_2_conv2')(attention2)
    attention2 = BN(attention2, 'attention2_2')
    attention = Add(name=name+'_add')([attention1,attention2])
    attention = Activation('sigmoid')(attention)
    attention = Repeat(repeat_list=[1, 1, 1, C])(attention)
    return attention

def ChannelWiseAttention(inputs,name):
    H, W, C = map(int, inputs.get_shape()[1:])
    attention = GlobalAveragePooling2D(name=name+'_GlobalAveragePooling2D')(inputs)
    attention = Dense(C / 4, activation='relu')(attention)
    attention = Dense(C, activation='sigmoid',activity_regularizer=l1_reg)(attention)
    attention = Reshape((1, 1, C),name=name+'_reshape')(attention)
    attention = Repeat(repeat_list=[1, H, W, 1],name=name+'_repeat')(attention)
    attention = Multiply(name=name + '_multiply')([attention, inputs])
    return attention

def SpatialAttention3D(inputs,name):
    k = 9
    H, W, D, C = map(int,inputs.get_shape()[1:])
    attention1 = tf.layers.conv3d(inputs=inputs, 
                   filters=int(C / 2),
                   kernel_size=(1,k,k),
                   strides=1,
                   padding=PADDING,
                   activation=lambda x, name=None: BN_Relu(x),
                   data_format=DATA_FORMAT,
                   name=name + "_1_conv1")
    # attention1 = Conv3D(int(C / 2), (1, k, k), padding='same', name=name+'_1_conv1')(inputs)
    # with tf.variable_scope(name + '_attention1_1') as scope:
    #     attention1 = BN_Relu(attention1)
    attention1 = tf.layers.conv3d(inputs=attention1, 
                   filters=1,
                   kernel_size=(k,1,1),
                   strides=1,
                   padding=PADDING,
                   activation=lambda x, name=None: BN_Relu(x),
                   data_format=DATA_FORMAT,
                   name=name + "_1_conv2")
    # attention1 = Conv3D(1, (k, 1, 1), padding='same', name=name + '_1_conv2')(attention1)
    # with tf.variable_scope(name + '_attention1_2') as scope:
    #     attention1 = BN_Relu(attention1)
    attention2 = tf.layers.conv3d(inputs=inputs, 
                   filters=int(C / 2),
                   kernel_size=(k,1,k),
                   strides=1,
                   padding=PADDING,
                   activation=lambda x, name=None: BN_Relu(x),
                   data_format=DATA_FORMAT,
                   name=name + "_2_conv1")
    attention2 = tf.layers.conv3d(inputs=attention2, 
                   filters=1,
                   kernel_size=(1,k,1),
                   strides=1,
                   padding=PADDING,
                   activation=lambda x, name=None: BN_Relu(x),
                   data_format=DATA_FORMAT,
                   name=name + "_2_conv2")
    # attention2 = Conv3D(int(C / 2), (k, 1, k), padding='same', name=name + '_2_conv1')(inputs)
    # with tf.variable_scope(name + '_attention2_1') as scope:
    #     attention2 = BN_Relu(attention2)
    # attention2 = Conv3D(1, (1, k, 1), padding='same', name=name + '_2_conv2')(attention2)
    # with tf.variable_scope(name + '_attention2_2') as scope:
    #     attention2 = BN_Relu(attention2)
    attention3 = tf.layers.conv3d(inputs=inputs, 
                   filters=int(C / 2),
                   kernel_size=(k,k,1),
                   strides=1,
                   padding=PADDING,
                   activation=lambda x, name=None: BN_Relu(x),
                   data_format=DATA_FORMAT,
                   name=name + "_3_conv1")
    attention3 = tf.layers.conv3d(inputs=attention3, 
                   filters=1,
                   kernel_size=(1,1,k),
                   strides=1,
                   padding=PADDING,
                   activation=lambda x, name=None: BN_Relu(x),
                   data_format=DATA_FORMAT,
                   name=name + "_3_conv2")
    # attention3 = Conv3D(int(C / 2), (k, k, 1), padding='same', name=name + '_3_conv1')(inputs)
    # with tf.variable_scope(name + '_attention3_1') as scope:
    #     attention3 = BN_Relu(attention3)
    # attention3 = Conv3D(1, (1, 1, k), padding='same', name=name + '_3_conv2')(attention3)
    # with tf.variable_scope(name + '_attention3_2') as scope:
    #     attention3 = BN_Relu(attention3)
    attention = tf.math.add_n([attention1,attention2,attention3], name=name+'_add')
    #attention = Add(name=name+'_add')([attention1,attention2,attention3])
    attention = tf.math.sigmoid(attention, name=name+'_sigmoid')
    #attention = Activation('sigmoid')(attention)
    attention = tf.tile(attention, [1, 1, 1, 1, C], name=name + '_repeat')
    #attention = Repeat(repeat_list=[1, 1, 1, 1, C], name=name + '_repeat')(attention)
    return attention

# def ChannelWiseAttention3D(inputs,name):
#     H, W, D, C = map(int, inputs.get_shape()[1:])
#     attention = GlobalAveragePooling3D(name=name+'_GlobalAveragePooling3D')(inputs)
#     attention = Dense(int(C / 4), activation='relu', name=name + '_dense_1')(attention)
#     attention = Dense(C, activation='sigmoid',activity_regularizer=l1_reg, name=name + '_dense_2')(attention)
#     attention = Reshape((1, 1, 1, C),name=name+'_reshape')(attention)
#     attention = Repeat(repeat_list=[1, H, W, D, 1],name=name+'_repeat')(attention)
#     attention = Multiply(name=name + '_multiply')([attention, inputs])
#     return attention

def ChannelWiseAttention3D(inputs,name):
    B, H, W, D, C = map(int, inputs.get_shape())
    attention = tf.reduce_mean(inputs, [1,2,3], name=name+'_GlobalAveragePooling3D')
    attention = tf.layers.dense(attention, int(C / 4), activation='relu', name=name + '_dense_1')
    attention = tf.layers.dense(attention, C, activation='sigmoid', name=name + '_dense_2')
    attention = tf.reshape(attention, (B, 1, 1, 1, C),name=name+'_reshape')
    attention = tf.tile(attention, [1, H, W, D, 1],name=name+'_repeat')
    attention = tf.math.multiply(attention, inputs, name=name + '_multiply')
    return attention
