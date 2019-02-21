#!/usr/bin/env python
# -*- coding: utf-8 -*-


from keras.applications import VGG16
from keras.preprocessing.image import ImageDataGenerator
from keras.preprocessing import image
from keras import models, layers
import shutil
import os, sys
import numpy as np
from keras.utils import to_categorical
from keras import optimizers

conv_base = VGG16(weights = 'imagenet', include_top = False, input_shape=(150,150,3))

base_dir = '/home/peichao/Desktop/python_experiment/all/cats_dogs_small'

validation_dir = os.path.join(base_dir, 'validation')
train_dir = os.path.join(base_dir, 'train')
test_dir = os.path.join(base_dir, 'test')

conv_base.trainable = False

network = models.Sequential()
network(conv_base)
network.add(layers.Flatten())
network.add(layers.Dense(256, activation = 'relu'))
network.add(layers.Dense(1, activation = 'relu'))
network.compile(optimizer = optimizers.RMSprop(2e-5), loss = 'binary_crossentropy', metrics=['acc'])


train_data = ImageDataGenerator(rescale=1./255, rotation_range=40, height_shift_range=0.2, width_shift_range=0.2, shear_range=0.2, zoom_range=0.2, horizontal_flip=True, fill_mode='nearest')
test_data = ImageDataGenerator(1./255)

train_generator = train_data.flow_from_directory(train_dir, target_size=(150,150), batch_size=20,class_mode='binary')
validation_generator = test_data.flow_from_directory(validation_dir, target_size=(150,150), batch_size=20, class_mode='bianry')

network.fit_generator(train_generator, steps_per_epoch=20, epochs=30, validation_data = validation_generator, validation_steps =20)