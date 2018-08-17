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
from keras.layers import Conv2D, SeparableConv2D, Reshape, multiply, Flatten
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

def cnnmodel1(input_shape):

    input = Input(shape=input_shape)
    x = conv2d_bn(input, 30, 3, 3, strides=(1, 1), padding='same')
    x = Flatten()(x)    
    x = Dense(10, activation='softmax')(x)

    inputs = input
    model = Model(inputs, x, name='model1')
    return model