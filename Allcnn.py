from __future__ import print_function
from __future__ import absolute_import

import warnings
import numpy as np
import tensorflow as tf

from keras.models import Model
from keras import layers
from keras.layers import Activation, Dropout
from keras.layers import Dense, Lambda
from keras.layers import Input
from keras.layers import BatchNormalization
from keras.layers import Conv2D, SeparableConv2D, Reshape, multiply
from keras.layers import MaxPooling2D
from keras.layers import AveragePooling2D
from keras.layers import GlobalAveragePooling2D
from keras.layers import GlobalMaxPooling2D
from keras.engine.topology import get_source_inputs
from keras.utils.layer_utils import convert_all_kernels_in_model
from keras.utils.data_utils import get_file
from keras import backend as K
from keras.applications.imagenet_utils import decode_predictions
from keras.applications.imagenet_utils import _obtain_input_shape
from keras.preprocessing import image
from keras import regularizers

def conv2d_bn(x,
              filters,
              num_row,
              num_col,
              padding='same',
              strides=(1, 1),
              name=None):
    """Utility function to apply conv + BN.
    Arguments:
        x: input tensor.
        filters: filters in `Conv2D`.
        num_row: height of the convolution kernel.
        num_col: width of the convolution kernel.
        padding: padding mode in `Conv2D`.
        strides: strides in `Conv2D`.
        name: name of the ops; will become `name + '_conv'`
            for the convolution and `name + '_bn'` for the
            batch norm layer.
    Returns:
        Output tensor after applying `Conv2D` and `BatchNormalization`.
    """
    if name is not None:
        bn_name = name + '_bn'
        conv_name = name + '_conv'
    else:
        bn_name = None
        conv_name = None
    if K.image_data_format() == 'channels_first':
        bn_axis = 1
    else:
        bn_axis = 3
    
    x = Conv2D(
        filters, (num_row, num_col),
        strides=strides,
        padding=padding,
        use_bias=False,        
        name=conv_name)(x)
    
    x = BatchNormalization(axis=bn_axis, scale=False, name=bn_name)(x)
    x = Activation('relu', name=name)(x)
    return x

def spconv2d_bn(x,
              filters,
              num_row,
              num_col,
              padding='same',
              strides=(1, 1),
              name=None):
    """Utility function to apply conv + BN.
    Arguments:
        x: input tensor.
        filters: filters in `Conv2D`.
        num_row: height of the convolution kernel.
        num_col: width of the convolution kernel.
        padding: padding mode in `Conv2D`.
        strides: strides in `Conv2D`.
        name: name of the ops; will become `name + '_conv'`
            for the convolution and `name + '_bn'` for the
            batch norm layer.
    Returns:
        Output tensor after applying `Conv2D` and `BatchNormalization`.
    """
    if name is not None:
        bn_name = name + '_bn'
        conv_name = name + '_conv'
    else:
        bn_name = None
        conv_name = None
    if K.image_data_format() == 'channels_first':
        bn_axis = 1
    else:
        bn_axis = 3
    
    
    x = SeparableConv2D(
        filters, (num_row, num_col),
        strides=strides,
        padding=padding,
        use_bias=False,        
        name=conv_name)(x)
    
    x = BatchNormalization(axis=bn_axis, scale=False, name=bn_name)(x)
    x = Activation('relu', name=name)(x)
    return x

def squeeze_excitation_layer(x, out_dim):
    '''
    SE module performs inter-channel weighting.
    '''
    squeeze = GlobalAveragePooling2D()(x)
    
    excitation = Dense(units=out_dim // 8)(squeeze)
    excitation = Activation('relu')(excitation)
    excitation = Dense(units=out_dim)(excitation)
    excitation = Activation('sigmoid')(excitation)
    excitation = Reshape((1,1,out_dim))(excitation)
    
    scale = multiply([x,excitation])
    
    return scale

def interplation(input,size):
    size = np.asarray(size)
    size = size[0,1]
    new_size = (size,size)
    return tf.image.resize_images(input,new_size,method=tf.image.ResizeMethod.BILINEAR)

def attention_block(input, filters):
    
    input = spconv2d_bn(input, filters, 3, 3, strides=(1, 1), padding='same')
    trunk_branch = spconv2d_bn(input, filters, 3, 3, strides=(1, 1), padding='same')    
    soft_branch = spconv2d_bn(input, filters, 3, 3, strides=(1, 1), padding='same')    
    soft_branch1 = MaxPooling2D((3, 3), strides=(2, 2),padding='same')(soft_branch)    
    soft_branch1 = spconv2d_bn(soft_branch1, filters, 3, 3, strides=(1, 1), padding='same')
    soft_branch2 = MaxPooling2D((3, 3), strides=(2, 2),padding='same')(soft_branch1)
    soft_branch2 = spconv2d_bn(soft_branch2, filters, 3, 3, strides=(1, 1), padding='same')
    soft_branch1 = Lambda(interplation,arguments={'size':[K.int_shape(soft_branch1)]})(soft_branch2)
    #soft_branch = UpSampling2D(size=(2,2),name=str(identifier)+'attention_up1')(soft_branch)    
    soft_branch1 = spconv2d_bn(soft_branch1, filters, 3, 3, strides=(1, 1), padding='same')
    soft_branch = Lambda(interplation,arguments={'size':[K.int_shape(soft_branch)]})(soft_branch1)
    #soft_branch = UpSampling2D(size=(2,2),name=str(identifier)+'attention_up2')(soft_branch)
    soft_branch = spconv2d_bn(soft_branch, filters, 3, 3, strides=(1, 1), padding='same')    
    soft_branch = Activation("sigmoid")(soft_branch)
    outputs = multiply([soft_branch,trunk_branch])
    outputs = layers.add([outputs,trunk_branch])

    return outputs

    

def all_cnn(input_shape):

    channel_axis = 3
    input = Input(shape=input_shape)

    x = spconv2d_bn(input, 96, 3, 3, strides=(1, 1), padding='same')
    # mixed 1: 28 x 28 x 384
    branch1x1 = conv2d_bn(x, 48, 1, 1, strides=(1,1))

    branch5x5 = conv2d_bn(x, 24, 1, 1)
    branch5x5 = spconv2d_bn(branch5x5, 48, 3, 3, strides=(1,1))

    branch3x3dbl = conv2d_bn(x, 24, 1, 1)
    branch3x3dbl = spconv2d_bn(branch3x3dbl, 24, 3, 3)
    branch3x3dbl = spconv2d_bn(branch3x3dbl, 48, 3, 3, strides=(1,1))

    branch_pool = AveragePooling2D((3, 3), strides=(1, 1), padding='same')(x)
    branch_pool = conv2d_bn(branch_pool, 48, 1, 1)
    x = layers.concatenate(
        [branch1x1, branch5x5, branch3x3dbl,branch_pool],
        axis=channel_axis,
        name='mixed1')
    #x = squeeze_excitation_layer(x, 192)
    
    x = conv2d_bn(x, 96 , 1 , 1)
    #mix2 = layers.add([x,mix])
    x = squeeze_excitation_layer(x, 96)
    #x = attention_block(x, 96)
    #x = layers.add([x,mix2])
    

    # mixed 2: 14 x 14 x 384
    branch1x1 = conv2d_bn(x, 48, 1, 1, strides=(2,2))

    branch5x5 = conv2d_bn(x, 24, 1, 1)
    branch5x5 = spconv2d_bn(branch5x5, 48, 3, 3, strides=(2,2))

    branch3x3dbl = conv2d_bn(x, 24, 1, 1)
    branch3x3dbl = spconv2d_bn(branch3x3dbl, 24, 3, 3)
    branch3x3dbl = spconv2d_bn(branch3x3dbl, 48, 3, 3, strides=(2,2))

    branch_pool = AveragePooling2D((3, 3), strides=(2, 2), padding='same')(x)
    branch_pool = conv2d_bn(branch_pool, 48, 1, 1)
    x = layers.concatenate(
        [branch1x1, branch5x5, branch3x3dbl, branch_pool],
        axis=channel_axis,
        name='mixed2')
    x = conv2d_bn(x, 96, 1, 1)
    #mix2 = spconv2d_bn(x, 96, 1, 1, strides=(2, 2))
    #mix2 = layers.add([mix, mix2])
    x = squeeze_excitation_layer(x, 96)
    #x = attention_block(x, 96)
    #x = spconv2d_bn(x, 96, 1, 1, strides=(2, 2))
    #x = layers.add([x, mix2])
    x = Dropout(0.5)(x)
    # mixed 3: 14 x 14 x 768
    branch1x1 = spconv2d_bn(x, 96, 1, 1, strides=(1,1))

    branch5x5 = conv2d_bn(x, 48, 1, 1)    
    branch5x5 = spconv2d_bn(branch5x5, 96, 3, 1, strides=(1,1))
    branch5x5 = spconv2d_bn(branch5x5, 96, 1, 3, strides=(1,1))

    branch3x3dbl = conv2d_bn(x, 48, 1, 1)    
    branch3x3dbl = spconv2d_bn(branch3x3dbl, 48, 3, 1)
    branch3x3dbl = spconv2d_bn(branch3x3dbl, 48, 1, 3)
    branch3x3dbl = spconv2d_bn(branch3x3dbl, 96, 3, 1, strides=(1,1))
    branch3x3dbl = spconv2d_bn(branch3x3dbl, 96, 1, 3, strides=(1,1))

    branch_pool = AveragePooling2D((3, 3), strides=(1, 1), padding='same')(x)
    branch_pool = conv2d_bn(branch_pool, 96, 1, 1)
    x = layers.concatenate(
        [branch1x1, branch5x5, branch3x3dbl, branch_pool],
        axis=channel_axis,
        name='mixed3')   
    x = conv2d_bn(x, 192, 1, 1)
    #x = spconv2d_bn(x, 192, 1, 1)
    #mix2 = layers.add([x, mix])
    x = squeeze_excitation_layer(x, 192)
    #x = attention_block(x, 192)
    #x = layers.add([x, mix2])
    # mixed 4: 14 x 14 x 192
    branch1x1 = conv2d_bn(x, 96, 1, 1, strides=(1,1))

    branch5x5 = conv2d_bn(x, 48, 1, 1)
    branch5x5 = spconv2d_bn(branch5x5, 96, 3, 1, strides=(1,1))
    branch5x5 = spconv2d_bn(branch5x5, 96, 1, 3, strides=(1,1))

    branch3x3dbl = conv2d_bn(x, 48, 1, 1)
    branch3x3dbl = spconv2d_bn(branch3x3dbl, 48, 3, 1)
    branch3x3dbl = spconv2d_bn(branch3x3dbl, 48, 1, 3)
    branch3x3dbl = spconv2d_bn(branch3x3dbl, 96, 3, 1, strides=(1,1))
    branch3x3dbl = spconv2d_bn(branch3x3dbl, 96, 1, 3, strides=(1,1))

    branch_pool = AveragePooling2D((3, 3), strides=(1, 1), padding='same')(x)
    branch_pool = conv2d_bn(branch_pool, 96, 1, 1)
    x = layers.concatenate(
        [branch1x1, branch5x5, branch3x3dbl, branch_pool],
        axis=channel_axis,
        name='mixed4')
    x = conv2d_bn(x, 192, 1, 1)
    #mix2 = layers.add([x, mix])
    x = squeeze_excitation_layer(x, 192)
    #x = attention_block(x, 192)
    #x = layers.add([x, mix2])
    # mixed 5: 7 x 7 x 192
    branch1x1 = conv2d_bn(x, 96, 1, 1, strides=(2,2))

    branch5x5 = conv2d_bn(x, 48, 1, 1)
    branch5x5 = spconv2d_bn(branch5x5, 96, 3, 1, strides=(1,1))
    branch5x5 = spconv2d_bn(branch5x5, 96, 1, 3, strides=(1,1))
    branch5x5 = spconv2d_bn(branch5x5, 96, 1, 1, strides=(2,2))

    branch3x3dbl = conv2d_bn(x, 48, 1, 1)
    branch3x3dbl = spconv2d_bn(branch3x3dbl, 48, 3, 1)
    branch3x3dbl = spconv2d_bn(branch3x3dbl, 48, 1, 3)
    branch3x3dbl = spconv2d_bn(branch3x3dbl, 96, 3, 3, strides=(2,2))

    branch_pool = AveragePooling2D((3, 3), strides=(2, 2), padding='same')(x)
    branch_pool = conv2d_bn(branch_pool, 96, 1, 1)
    x = layers.concatenate(
        [branch1x1, branch5x5, branch3x3dbl, branch_pool],
        axis=channel_axis,
        name='mixed5')
    x = conv2d_bn(x, 192, 1, 1)
    #mix2 = spconv2d_bn(x, 192, 1, 1, strides=(2, 2))
    #mix2 = layers.add([mix, mix2])
    x = squeeze_excitation_layer(x, 192)
    #x = attention_block(x, 192)
    #x = spconv2d_bn(x, 192, 1, 1, strides=(2, 2))
    #x = layers.add([x, mix2])    
    x = Dropout(0.5)(x)
    
    
    x = spconv2d_bn(x, 192, 3, 3, strides=(1,1), padding='same')    
    x = squeeze_excitation_layer(x, 192)   

    x = spconv2d_bn(x, 192, 1, 1, strides=(1,1), padding='same')    
    x = squeeze_excitation_layer(x, 192)   
    
    x = conv2d_bn(x, 10, 1, 1, strides=(1,1), padding='same')
    x = GlobalAveragePooling2D(data_format='channels_last', name='avg_pool')(x)
    x = Activation('softmax')(x)

    inputs = input
    model = Model(inputs, x, name='all_cnn')
    return model