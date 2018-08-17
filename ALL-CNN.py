
# coding: utf-8

from __future__ import print_function
import keras
from keras.datasets import cifar10
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Conv2D, MaxPooling2D,GlobalAveragePooling2D,GlobalAveragePooling1D
from keras.layers.advanced_activations import LeakyReLU
from keras.layers.normalization import BatchNormalization
from keras.callbacks import ModelCheckpoint
from keras.initializers import RandomUniform
from keras.utils import plot_model
import os
import pickle
import numpy as np
import tensorflow as tf

from img_process import *
from lsuv_init import LSUVinit

flags = tf.app.flags
flags.DEFINE_integer("epochs", 200, "Epoch to train[200]")
flags.DEFINE_float("learning_rate", 0.001, "Learning rate of for adam[0.001] ")
flags.DEFINE_boolean("load_model", False, "使用分類器時是否載入權重")
FLAGS = flags.FLAGS

num_classes = 10
batch_size = 100
(x_train, y_train), (x_test, y_test) = cifar10.load_data()

x_train = x_train.astype('float32')
x_test = x_test.astype('float32')
x_train /= 255
x_test /= 255
# Convert class vectors to binary class matrices.
y_train = keras.utils.to_categorical(y_train, num_classes)
y_test = keras.utils.to_categorical(y_test, num_classes)

h_train = horizontal(x_train)
x_train=np.concatenate((x_train,h_train),axis=0)
y_train=np.concatenate((y_train,y_train),axis=0)
x_train=re_scale(x_train)
x_test=re_scale(x_test)#重塑至[0,1]
# The data, shuffled and split between train and test sets:
print('x_train shape:', x_train.shape)
print(x_train.shape[0], 'train samples')
print(x_test.shape[0], 'test samples')


model = Sequential()

model.add(Conv2D(96, (2, 2), padding='same', 
                 input_shape=x_train.shape[1:] ))
model.add(BatchNormalization())
model.add(LeakyReLU())
model.add(Conv2D(96,  (2, 2), padding='same'))
model.add(BatchNormalization())
model.add(LeakyReLU())
model.add(Conv2D(96,  (2, 2), strides=(2,2), padding='same'))
model.add(BatchNormalization())
model.add(LeakyReLU())

model.add(Conv2D(192,  (2, 2), padding='same'))
model.add(BatchNormalization())
model.add(LeakyReLU())
#model.add(Dropout(0.1))
model.add(Conv2D(192,  (2, 2), padding='same'))
model.add(BatchNormalization())
model.add(LeakyReLU())
#model.add(Dropout(0.1))
model.add(Conv2D(192,  (2, 2), strides=(2,2), padding='same'))
model.add(BatchNormalization())
model.add(LeakyReLU())
model.add(Dropout(0.1))

model.add(Conv2D(288,  (2, 2), padding='same'))
model.add(BatchNormalization())
model.add(LeakyReLU())
#model.add(Dropout(0.2))
model.add(Conv2D(288, (2, 2), padding='same'))
model.add(BatchNormalization())
model.add(LeakyReLU())
#model.add(Dropout(0.2))
model.add(Conv2D(288,  (2, 2), strides=(2,2), padding='same'))
model.add(BatchNormalization())
model.add(LeakyReLU())
model.add(Dropout(0.2))

model.add(Conv2D(384,  (2, 2), padding='same'))
model.add(BatchNormalization())
model.add(LeakyReLU())
#model.add(Dropout(0.3))
model.add(Conv2D(384,  (2, 2), padding='same'))
model.add(BatchNormalization())
model.add(LeakyReLU())
#model.add(Dropout(0.3))
model.add(Conv2D(384, (2, 2), strides=(2,2), padding='same'))
model.add(BatchNormalization())
model.add(LeakyReLU())
model.add(Dropout(0.3))

model.add(Conv2D(480,  (2, 2), padding='same'))
model.add(BatchNormalization())
model.add(LeakyReLU())
#model.add(Dropout(0.4))
model.add(Conv2D(480,  (2, 2), padding='same'))
model.add(BatchNormalization())
model.add(LeakyReLU())
#model.add(Dropout(0.4))
model.add(Conv2D(480,  (2, 2), strides=(2,2), padding='same'))
model.add(BatchNormalization())
model.add(LeakyReLU())
model.add(Dropout(0.4))

model.add(Conv2D(576,  (2, 2), padding='same'))
model.add(BatchNormalization())
model.add(LeakyReLU())
model.add(Dropout(0.5))
model.add(Conv2D(576, (1, 1), padding='same'))
model.add(BatchNormalization())
model.add(LeakyReLU())
model.add(Dropout(0.5))

#model.add(Flatten())
#model.add(Dense(num_classes))
model.add(Conv2D(10, (1, 1), padding='same'))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(GlobalAveragePooling2D(data_format='channels_last'))
model.add(BatchNormalization())
model.add(Activation('softmax'))

model = LSUVinit(model,x_train[:batch_size,:,:,:])
'''
model = Sequential()
model.add(Conv2D(96, (3, 3), padding='same',input_shape=x_train.shape[1:]))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(Conv2D(96, (3, 3), padding='same'))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(Conv2D(96, (3, 3), padding='same',strides=(2,2)))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(Dropout(0.5))

model.add(Conv2D(192, (3, 3), padding='same'))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(Conv2D(192, (3, 3), padding='same'))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(Conv2D(192, (3, 3), padding='same',strides=(2,2)))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(Dropout(0.5))

model.add(Conv2D(192, (3, 3), padding='valid'))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(Conv2D(192, (1, 1), padding='same'))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(Conv2D(10, (1, 1), padding='same'))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(GlobalAveragePooling2D(data_format='channels_last'))
model.add(BatchNormalization())
model.add(Activation('softmax'))
'''
model.summary()
plot_model(model, to_file='model.png')
# train the model
decay_rate = FLAGS.learning_rate / FLAGS.epochs
opt = keras.optimizers.Adam(lr=FLAGS.learning_rate,decay=decay_rate) 
model.compile(loss='categorical_crossentropy',optimizer=opt,metrics=['accuracy'])

checkpoint = ModelCheckpoint('./checkpoint/weights.{epoch:02d}-{val_loss:.2f}.hdf5', monitor='val_loss', verbose=0,save_best_only=False, save_weights_only=False, mode='auto', period=10)
callbacks_list = [checkpoint]
if (FLAGS.load_model):
	model.load_weights('./checkpoint/cnn_weights.h5')
model.fit(x_train, y_train,batch_size,epochs=FLAGS.epochs,validation_data=(x_test, y_test),verbose=1,callbacks=callbacks_list)
model.save_weights('./checkpoint/cnn_weights.h5')
#model.save_weights('./cnn_model/{}_{}_{}.h5'.format(config.choose_mode,self.img_proce,self.crop_size))
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

for i in range(10):
    right=right+matrix[i][i]

    
print("total:",len(result))
print("正確句數：",right)
print("正確率:",right/len(result)*100)
