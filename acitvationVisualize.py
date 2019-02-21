#!/usr/bin/env python
# -*- coding: utf-8 -*-

from keras import models,layers
from keras.layers import Embedding
from keras.preprocessing import image
from keras.preprocessing.image import ImageDataGenerator
import numpy as np
import os, sys
from keras.models import load_model
import matplotlib.pyplot as plt

'''
base_dir = '/home/peichao/Desktop/python_experiment/all/train/cat.1700.jpg'
img = image.load_img(base_dir, target_size=(150,150))
image_tensor = image.img_to_array(img)
image_tensor = np.expand_dims(image_tensor, axis=0)
image_tensor /= 255.

'''
model = models.Sequential()
model = load_model('cats_and_dogs_small_2.h5')


# exchange to array and normalization.
base_dir = '/home/peichao/Desktop/python_experiment/all/train/cat.1700.jpg'
img = image.load_img(base_dir, target_size=(150,150))
image_tensor = image.img_to_array(img)
image_tensor = np.expand_dims(image_tensor, axis=0)
image_tensor /= 255.

# extract output of previous 8 layers
layer_outputs = [layer.output for layer in model.layers[:8]]
# create a new model include input and output
activation_model = models.Model(inputs=model.input, outputs=layer_outputs)

# return list of numpy array, activation of every layer correspending to array of numpy
activations = activation_model.predict(image_tensor)

#print(activatons[3].shape)
#first_layer_activation = activations[0]

layer_names = []
for layer in model.layers[:8]:
    layer_names.append(layer.name)

image_per_row = 16

for layer_name, layer_activation in zip(layer_names, activations):   #show feature diagram
    n_feature = layer_activation.shape[-1]    # show number of features of feature diagram(Actually show channels)
    #print(n_feature)
    size = layer_activation.shape[1]  # show 4
    #print(size)
    n_cols = n_feature // image_per_row

    display_grid = np.zeros((size*n_cols, image_per_row*size))
    for col in range(n_cols):
        for row in range(image_per_row):
            channel_image = layer_activation[0,:,:,col*image_per_row+row]
            channel_image -=channel_image.mean()
            channel_image /= channel_image.std()
            channel_image *= 64
            channel_image += 128
            channel_image = np.clip(channel_image,0,255).astype('uint8')
            display_grid[col*size:(col+1)*size, row*size:(row+1)*size]


