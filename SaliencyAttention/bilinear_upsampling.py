from keras import backend as K
from keras.engine import Layer
from keras import initializers
from keras.utils import conv_utils
from keras.engine import InputSpec
import tensorflow as tf
import numpy as np
import config

class BilinearUpsampling(Layer):
    """Just a simple bilinear upsampling layer. Works only with TF.
       Args:
           upsampling: tuple of 2 numbers > 0. The upsampling ratio for h and w
           output_size: used instead of upsampling arg if passed!
    """

    def __init__(self, upsampling=(2, 2), output_size=None, data_format=None, **kwargs):

        super(BilinearUpsampling, self).__init__(**kwargs)

        self.data_format = conv_utils.normalize_data_format(data_format)
        self.input_spec = InputSpec(ndim=4)
        if output_size:
            self.output_size = conv_utils.normalize_tuple(
                output_size, 2, 'output_size')
            self.upsampling = None
        else:
            self.output_size = None
            self.upsampling = conv_utils.normalize_tuple(
                upsampling, 2, 'upsampling')

    def compute_output_shape(self, input_shape):
        if self.upsampling:
            height = self.upsampling[0] * \
                input_shape[1] if input_shape[1] is not None else None
            width = self.upsampling[1] * \
                input_shape[2] if input_shape[2] is not None else None
        else:
            height = self.output_size[0]
            width = self.output_size[1]
        return (input_shape[0],
                height,
                width,
                input_shape[3])

    def call(self, inputs):
        if self.upsampling:
            return K.tf.image.resize_bilinear(inputs, (int(inputs.shape[1] * self.upsampling[0]),
                                                       int(inputs.shape[2] * self.upsampling[1])),
                                              align_corners=True)
        else:
            return K.tf.image.resize_bilinear(inputs, (self.output_size[0],
                                                       self.output_size[1]),
                                              align_corners=True)

    def get_config(self):
        config = {'upsampling': self.upsampling,
                  'output_size': self.output_size,
                  'data_format': self.data_format}
        base_config = super(BilinearUpsampling, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

# class BilinearUpsampling3D(Layer):
#     """Just a simple bilinear upsampling layer. Works only with TF.
#        Args:
#            upsampling: tuple of 3 numbers > 0. The upsampling ratio for h, w and d
#     """

#     def __init__(self, upsampling=(2, 2, 2), **kwargs):

#         super(BilinearUpsampling3D, self).__init__(**kwargs)

#         self.upsampling = upsampling

#     def build(self, input_shape):
#         assert len(input_shape) == 5

#         self.input_h = input_shape[1]
#         self.input_w = input_shape[2]
#         self.input_d = input_shape[3]
#         self.input_c = input_shape[4]

#         #Transformation matrix
#         self.W1 = self.add_weight(shape=[self.upsampling[0], self.upsampling[1], self.upsampling[2], self.input_c, self.input_c],
#                                 initializer=initializers.Ones(),
#                                 trainable=False,
#                                 name='W1')

#         self.W2 = self.add_weight(shape=[self.upsampling[0], self.upsampling[1], self.upsampling[2], self.input_c, self.input_c],
#                                 initializer=initializers.Constant(1.0 / (self.upsampling[0] * self.upsampling[1] * self.upsampling[2])),
#                                 trainable=False,
#                                 name='W2')

#         self.built = True

#     def compute_output_shape(self, input_shape):
#         output_shape = list(input_shape)

#         output_shape[1] = conv_utils.deconv_length(output_shape[1], self.upsampling[0], self.upsampling[0], 'same')
#         output_shape[2] = conv_utils.deconv_length(output_shape[2], self.upsampling[1], self.upsampling[1], 'same')
#         output_shape[3] = conv_utils.deconv_length(output_shape[3], self.upsampling[2], self.upsampling[2], 'same')

#         return tuple(output_shape)

#     def call(self, inputs, training=None):
#         input_shape = K.shape(inputs)

#         batch_size = input_shape[0]

#         #Infer the dynamic output shape
#         out_h = conv_utils.deconv_length(self.input_h, self.upsampling[0], self.upsampling[0], 'same')
#         out_w = conv_utils.deconv_length(self.input_w, self.upsampling[1], self.upsampling[1], 'same')
#         out_d = conv_utils.deconv_length(self.input_d, self.upsampling[2], self.upsampling[2], 'same')

#         output_shape = (batch_size, out_h, out_w, out_d, self.input_c)

#         outputs = tf.nn.conv3d_transpose(inputs, self.W1, output_shape, strides=self.upsampling[0],
#                                         padding='SAME', data_format='NDHWC')

#         return tf.nn.conv3d(outputs,
#                             self.W2,
#                             strides=[1, 1, 1, 1, 1],
#                             padding='SAME')

#     def get_config(self):
#         config = {'upsampling': self.upsampling}
#         base_config = super(BilinearUpsampling3D, self).get_config()
#         return dict(list(base_config.items()) + list(config.items()))

def BilinearUpsampling3D(input_tensor, upsamplescale, name):
    H, W, D, C = map(int, input_tensor.get_shape()[1:])
    deconv = tf.nn.conv3d_transpose(input_tensor, filter=tf.constant(np.ones([upsamplescale,upsamplescale,upsamplescale,C,C], np.float32)), output_shape=[config.BATCH_SIZE, H * upsamplescale, W * upsamplescale, D * upsamplescale, C],
                                strides=[1, upsamplescale, upsamplescale, upsamplescale, 1],
                                padding="SAME", name=name + '_UpsampleDeconv')
    smooth5d = tf.constant(np.ones([upsamplescale,upsamplescale,upsamplescale,C,C],dtype='float32')/np.float32(upsamplescale)/np.float32(upsamplescale)/np.float32(upsamplescale), name=name + '_Upsample'+str(upsamplescale))
    #print('Upsample', upsamplescale)
    return tf.nn.conv3d(input=deconv,
                 filter=smooth5d,
                 strides=[1, 1, 1, 1, 1],
                 padding='SAME',
                 name=name + '_UpsampleSmooth'+str(upsamplescale))
