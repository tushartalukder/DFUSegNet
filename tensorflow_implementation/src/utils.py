# src/utils.py

import os
import tensorflow as tf
import numpy as np
from scipy.ndimage import binary_erosion, generate_binary_structure
from scipy.ndimage import distance_transform_edt
from src.model_parts import conv2d_bn, conv2d, repeat_elem

def hed(x):
    sobel_edges = tf.image.sobel_edges(x)
    magnitude = tf.sqrt(tf.reduce_sum(tf.square(sobel_edges), axis=-1))
    magnitude = tf.maximum(magnitude, 1e-10)
    magnitude = tf.image.per_image_standardization(magnitude)
    return magnitude

def boundary_enhancer(u, d):
    filters = u.shape[-1]
    d = conv2d_bn(d, filters, 1, 1, activation='relu', padding='same')
    d = tf.keras.layers.UpSampling2D(size=(2, 2), interpolation='bilinear')(d)
    
    u_sobel = hed(u)
    d_sobel = hed(d)
    ud_sobel = u_sobel * d_sobel
    
    conv1 = conv2d_bn(ud_sobel, filters, 3, 3, activation='relu', padding='same')
    conv1 = conv2d(conv1, filters // 2, 3, 3, activation='relu', padding='same')
    boundary_map = conv2d(conv1, 1, 3, 3, activation='sigmoid', dilation_rate=1, padding='same')
    
    mapping = repeat_elem(boundary_map, filters)
    out = tf.keras.layers.add([u, mapping])
    return out, boundary_map

def summarize_performance(epoch, g_model, save_dir):
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    filename = f'gmodel_{epoch+1:06d}.h5'
    filepath = os.path.join(save_dir, filename)
    g_model.save(filepath)
    print(f'>Saved model to: {filepath}')

def remove_prev_performance(epoch, save_dir):
    if epoch == 0: return
    filename = f'gmodel_{epoch:06d}.h5'
    filepath = os.path.join(save_dir, filename)
    if os.path.exists(filepath):
        os.remove(filepath)
        print(f'>Removed old model: {filepath}')