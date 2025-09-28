# src/losses.py

import tensorflow as tf
from tensorflow.keras import backend as K

def bce_loss(y_true, y_pred):
    bce = tf.keras.losses.BinaryCrossentropy()
    return bce(y_true, y_pred)

def dice_loss(y_true, y_pred, smooth=1e-6):
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    return 1 - (2. * intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)

def bce_dice_loss(y_true, y_pred, dice_weight=1.0):
    bce = bce_loss(y_true, y_pred)
    dice = dice_loss(y_true, y_pred)
    return 2 * bce + dice_weight * dice

def focal_loss(y_true, y_pred, gamma=2.0, alpha=0.25):
    y_true = tf.cast(y_true, tf.float32)
    y_pred = tf.cast(y_pred, tf.float32)
    bce = K.binary_crossentropy(y_true, y_pred)
    p_t = y_true * y_pred + (1 - y_true) * (1 - y_pred)
    fl = K.pow(1 - p_t, gamma) * bce
    return K.mean(fl)

def structure_loss(pred, mask):
    weit = 1 + 5 * tf.abs(tf.nn.avg_pool2d(mask, ksize=31, strides=1, padding='SAME') - mask)
    wbce = tf.keras.losses.binary_crossentropy(mask, pred, from_logits=False) # from_logits=False if sigmoid is applied
    wbce = tf.expand_dims(wbce, axis=-1)
    wbce = tf.reduce_sum(weit * wbce, axis=(1, 2)) / tf.reduce_sum(weit, axis=(1, 2))
    
    inter = tf.reduce_sum((pred * mask) * weit, axis=(1, 2, 3))
    union = tf.reduce_sum((pred + mask) * weit, axis=(1, 2, 3))
    wiou = 1 - (inter + 1) / (union - inter + 1)
    
    return tf.reduce_mean(wbce + wiou)