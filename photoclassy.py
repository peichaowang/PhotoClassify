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
import matplotlib.pyplot as plt

origianl_dir = '/home/peichao/Desktop/python_experiment/all/train'
base_dir = '/home/peichao/Desktop/python_experiment/exercise'
os.mkdir(base_dir)

train_dir = os.path.join(base_dir, 'train')
os.mkdir(train_dir)
validation_dir = os.path.join(base_dir, 'validation')
os.mkdir(validation_dir)
test_dir = os.path.join(base_dir, 'test')

train_cat_dir = os.path.join(train_dir, 'cat')
os.mkdir(train_cat_dir)
train_dog_dir = os.path.join(train_dir, 'dog')
os.mkdir(train_dog_dir)

validation_cat_dir = os.path.join(validation_dir, 'cat')
os.mkdir(validation_cat_dir)
validation_dog_dir = os.path.join(validation_dir, 'dog')
os.mkdir(train_dog_dir)

test_cat_dir = os.path.join(test_dir, 'cat')
os.mkdir(test_cat_dir)
test_dog_dir = os.path.join(test_dir, 'dog')
os.mkdir(test_dog_dir)

fname = ['cat.{}.jpg'.format(i) for i in range(1000)]
for name in fname:
	src = os.path.join(origianl_dir, name)
	dst = os.path.join(train_cat_dir, name)
	shutil.copyfile(src,dst)

fname = ['cat.{}.jpg'.format(i) for i in range(1000,1500)]
for name in fname:
	src = os.path.join(origianl_dir, name)
	dst = os.path.join(validation_cat_dir, name)
	shutil.copyfile(src,dst)


fname = ['cat.{}.jpg'.format(i) for i in range(1500,2000)]
for name in fname:
	src = os.path.join(origianl_dir, name)
	dst = os.path.join(test_cat_dir, name)
	shutil.copyfile(src,dst)

fname = ['dog.{}.jpg'.format(i) for i in range(1000)]
for name in fname:
	src = os.path.join(origianl_dir, name)
	dst = os.path.join(train_dog_dir, name)
	shutil.copyfile(src,dst)

fname = ['dog.{}.jpg'.format(i) for i in range(1000,1500)]
for name in fname:
	src = os.path.join(origianl_dir, name)
	dst = os.path.join(validation_dog_dir, name)
	shutil.copyfile(src,dst)


fname = ['dog.{}.jpg'.format(i) for i in range(1500,2000)]
for name in fname:
	src = os.path.join(origianl_dir, name)
	dst = os.path.join(test_dog_dir, name)
	shutil.copyfile(src,dst)

network = models.Sequential()
network.add(layers.Conv2D(32,(3,3),activation = 'relu' ,input_shape = (150,150,3)))
network.add(layers.MaxPool2D(2,2))
network.add(layers.Conv2D(64,(3,3),activation = 'relu'))
network.add(layers.MaxPool2D(2,2))
network.add(layers.Conv2D(128,(3,3),activation = 'relu'))
network.add(layers.MaxPool2D(2,2))
network.add(layers.Conv2D(128,(3,3),activation = 'relu'))
network.add(layers.MaxPool2D(2,2))
network.add(layers.Flatten())
network.add(layers.Dropout(0.5))
network.add(layers.Dense(512, activation = 'relu'))
network.add(layers.Dense(1, activation = 'relu'))
network.compile(optimizer = optimizers.RMSprop(lr=1e-4), loss = 'binary_crossentropy', metrics=['acc'])

train_data = ImageDataGenerator(rescale=1./255, rotation_range=40, height_shift_range=0.2, width_shift_range=0.2, shear_range=0.2, zoom_range=0.2,horizontal_flip=True)
test_data = ImageDataGenerator(rescale=1./255)

train_generator = train_data.flow_from_directory(train_dir, target_size=(150,150), batch_size=20, class_mode='binary')
validation_generator = test_data.flow_from_directory(validation_dir, target_size=(150,150), batch_size=20, class_mode='binary')

history = network.fit_generator(train_generator, steps_per_epoch = 20, epochs=20, validation_data = validation_generator, validation_steps=20)





