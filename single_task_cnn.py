
# coding: utf-8

from __future__ import print_function
import keras
import tensorflow as tf
from keras.datasets import cifar10, mnist
from keras.models import Sequential, Model
from keras.layers import *
from keras.layers.advanced_activations import LeakyReLU
from keras.layers.normalization import BatchNormalization
from keras.callbacks import ModelCheckpoint, TensorBoard
from keras.initializers import RandomUniform
from keras.utils import plot_model
from keras.backend.tensorflow_backend import _to_tensor
from keras import backend as K
from keras.utils.generic_utils import get_custom_objects
import cv2
import os
import pickle
import numpy as np
import tensorflow as tf
import math
from skimage.feature import hog

from img_process import *
from lsuv_init import LSUVinit
from custom_act import cigmoid,ctanh
from layers import ConvOffset2D
from inceptionv3 import InceptionV3
from Allcnn import all_cnn
from fcmodel import model1
from cnnmodel1 import cnnmodel1
from cnnmodel2 import cnnmodel2
from cnnmodel3 import cnnmodel3
import mnist_reader

flags = tf.app.flags
flags.DEFINE_integer("epochs", 50, "Epoch to train")
flags.DEFINE_float("learning_rate", 0.001, "Learning rate of for adam[0.001] ")
flags.DEFINE_boolean("load_model", False, "loading model weights")
flags.DEFINE_boolean("multi_task", False, "using pretraning classifier weights")
flags.DEFINE_integer("datasets", 0, "0:using MNIST datasets,1:using fashionMNIST datasets,2:using notMNIST datasets")
flags.DEFINE_integer("d_component",1,"number of component")
flags.DEFINE_integer("imgsize",21,"size of images")
flags.DEFINE_integer("model",2,"number of models")
FLAGS = flags.FLAGS
num_classes = 10
batch_size = 64
component_1=96//FLAGS.d_component
component_2=192//FLAGS.d_component

if(FLAGS.datasets == 0):
    data_class_name=['0','1','2','3','4','5','6','7','8','9']    
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
if(FLAGS.datasets == 1):
    data_class_name=['T-shirt','Trouser','Pullover','Dress','Coat','Sandal','Shirt','Sneaker','Bag','Ankle boot']
    x_train, y_train = mnist_reader.load_mnist('data/fashion', kind='train')
    x_test, y_test = mnist_reader.load_mnist('data/fashion', kind='t10k')
if(FLAGS.datasets == 2):
    data_class_name=['A','B','C','D','E','F','G','H','I','J']
    x_train, y_train = mnist_reader.load_mnist('data/notMNIST', kind='train')
    x_test, y_test = mnist_reader.load_mnist('data/notMNIST', kind='t10k') 

def hogimg(x):
    img=[]
    for i in range(len(x)):
        temp_fd, temp_img = hog(x[i,:,:], orientations=3, pixels_per_cell=(3, 3), transform_sqrt=True,
                        cells_per_block=(1, 1), visualise=True)
        img.append(temp_img)
    img = np.asarray(img)
    img = img.reshape(x.shape[0],x.shape[1],x.shape[1],1)
    return img

#x_train=re_scale(x_train,-1,1)
#x_test=re_scale(x_test,-1,1)#重塑至[0,1]
x_train = x_train.astype('float32') / 255.
x_test = x_test.astype('float32') / 255.
x_train = np.reshape(x_train,[-1,28,28,1])
x_test = np.reshape(x_test,[-1,28,28,1])
x_train = x_train.astype('float32')
x_test = x_test.astype('float32')

x_train = img_resize(x_train, FLAGS.imgsize)
x_test = img_resize(x_test, FLAGS.imgsize)

#x_train = hogimg(x_train)
#x_test = hogimg(x_test)

x_train=i2b(x_train)
x_test=i2b(x_test)#image to binary

#x_train_noisy = x_train + np.random.normal(loc=0.0, scale=0.25, size=x_train.shape)
#x_train = np.concatenate((x_train, x_train_noisy), axis=0) 

# Convert class vectors to binary class matrices.
y_train = keras.utils.to_categorical(y_train, num_classes)
y_test = keras.utils.to_categorical(y_test, num_classes)

#y_train = np.concatenate((y_train, y_train), axis=0)

# The data, shuffled and split between train and test sets:
print('x_train shape:', x_train.shape)
print(y_train.shape)
print(x_train.shape[0], 'train samples')
print(x_test.shape[0], 'test samples')

def fractional_max_pool(inputs, ratio):
	return tf.nn.fractional_max_pool(value=inputs, pooling_ratio=ratio, name='fraction-max-pooling')[0]
#conv1_1_pool = Lambda(fractional_max_pool, arguments={'ratio': [1.0, math.sqrt(2), math.sqrt(2), 1.0]})(conv1_1)

def nor(x):
    x = ((x-K.min(x))*2/(K.max(x)-K.min(x)))-1
    return x
'''
model = Sequential()
#model.add(Conv2D(component_1, (3, 3), padding='same', input_shape=x_train.shape[1:]))
model.add(SeparableConv2D(component_1, (3, 3), padding='same',trainable=False ,input_shape=x_train.shape[1:]))
#model.add(LocallyConnected2D(component_1, (3, 3), input_shape=x_train.shape[1:]))
#model.add(Lambda(fractional_max_pool, arguments={'ratio': [1.0, math.sqrt(2), math.sqrt(2), 1.0]}))
model.add(BatchNormalization())
model.add(Activation('relu'))

#model.add(ConvOffset2D(component_1))
#model.add(Conv2D(component_1, (3, 3), padding='same'))
model.add(SeparableConv2D(component_1, (3, 3), padding='same'))
#model.add(LocallyConnected2D(component_1, (3, 3)))
model.add(BatchNormalization())
model.add(Activation('relu'))

#model.add(ConvOffset2D(component_1))
#model.add(Conv2D(component_1, (3, 3), padding='same',strides=(2,2)))
model.add(SeparableConv2D(component_1, (3, 3), padding='same',strides=(2,2)))
#model.add(LocallyConnected2D(component_1, (3, 3), strides=(2,2)))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(Dropout(0.5))

#model.add(ConvOffset2D(component_1))
#model.add(Conv2D(component_2, (3, 3), padding='same'))
model.add(SeparableConv2D(component_2, (3, 3), padding='same'))
#model.add(LocallyConnected2D(component_2, (3, 3)))
model.add(BatchNormalization())
model.add(Activation('relu'))

#model.add(ConvOffset2D(component_2))
#model.add(Conv2D(component_2, (3, 3), padding='same'))
model.add(SeparableConv2D(component_2, (3, 3), padding='same'))
#model.add(LocallyConnected2D(component_2, (3, 3)))
model.add(BatchNormalization())
model.add(Activation('relu'))

#model.add(ConvOffset2D(component_2))
#model.add(Conv2D(component_2, (3, 3), padding='same',strides=(2,2)))
model.add(SeparableConv2D(component_2, (3, 3), padding='same',strides=(2,2)))
#model.add(LocallyConnected2D(component_2, (3, 3), strides=(2,2)))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(Dropout(0.5))

#model.add(ConvOffset2D(component_2))
#model.add(Conv2D(component_2, (3, 3), padding='valid'))
#model.add(SeparableConv2D(component_2, (3, 3), padding='valid'))
model.add(LocallyConnected2D(component_2, (3, 3), padding='valid'))
model.add(BatchNormalization())
model.add(Activation('relu'))

#model.add(ConvOffset2D(component_2))
#model.add(Conv2D(component_2, (1, 1), padding='same'))
#model.add(SeparableConv2D(component_2, (1, 1), padding='same'))
model.add(LocallyConnected2D(component_2, (3, 3)))
model.add(BatchNormalization())
model.add(Activation('relu'))
#model.add(Lambda(nor))
#model.add(Activation(cigmoid))

model.add(Conv2D(10, (1, 1), padding='same'))
#model.add(SeparableConv2D(10, (1, 1), padding='same'))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(GlobalAveragePooling2D(data_format='channels_last'))
model.add(BatchNormalization())
model.add(Activation('softmax'))
'''

if(FLAGS.model == 0):
    model = model1(input_shape=x_train.shape[1:])    
if(FLAGS.model == 1):
    model = cnnmodel1(input_shape=x_train.shape[1:])    
if(FLAGS.model == 2):
    model = cnnmodel2(input_shape=x_train.shape[1:])    
if(FLAGS.model == 3):
    model = cnnmodel3(input_shape=x_train.shape[1:])
if(FLAGS.model == 4):
    model = all_cnn(input_shape=x_train.shape[1:])

#model = LSUVinit(model, x_train[:batch_size,:,:,:])

#model = InceptionV3(include_top=True,weights=None,input_shape=x_train.shape[1:])

model.summary()
opts = tf.profiler.ProfileOptionBuilder.float_operation()    
flops = tf.profiler.profile(tf.get_default_graph(), run_meta=tf.RunMetadata(), cmd='op', options=opts)
print('TF stats gives',flops.total_float_ops)
#plot_model(model, to_file='model.png')
'''
if(FLAGS.multi_task==True):
    for i in range(27):
        w = np.load('./layer_weight/weights{}.00.npy'.format(i))
'''
        


# train the model  

decay_rate = FLAGS.learning_rate / FLAGS.epochs
opt = keras.optimizers.Adam(lr=FLAGS.learning_rate,decay=decay_rate) 
model.compile(loss='categorical_crossentropy',optimizer=opt,metrics=['accuracy'])  
print('----dataset:', FLAGS.datasets, 'imgsize:', FLAGS.imgsize, 'models:', FLAGS.model,'----')
if (FLAGS.load_model):
    model.load_weights('./checkpoint/learning_decay_weights.h5')
model.fit(x_train, y_train,batch_size,epochs=FLAGS.epochs,validation_data=(x_test, y_test),verbose=0)    
if (FLAGS.load_model == False):
    model.save_weights('./checkpoint/learning_decay_weights.h5')
model.save_weights('./checkpoint/cnn_weights{}.h5'.format(FLAGS.datasets))

result = model.predict(x_test)

matrix = np.zeros((10, 10))
right=0

for i, (Target, Label) in enumerate( zip(y_test, result) ) :  ### i-th label
    m = np.max(Label)
    for j, value in enumerate(Label) :  ### find max value and position
        if value == m :
            for k, num in enumerate(Target) :  ### find test calss
                if num == 1 :
                    matrix[k][j] += 1
                    break  # end of for k
            break  # end of for j

print('classification result')
for i in range(10):
    print('{}:{:.2f}% '.format(data_class_name[i],(matrix[i][i]/np.sum(matrix[i]))*100),end="")
    if i==4:print()
    right=right+matrix[i][i]

print("total:",len(result))
print("correct number：",right)
print("correct rate：{:.2f}%".format(right/len(result)*100))