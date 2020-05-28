import tensorflow as tf
import tensorflow.keras.backend as K
import numpy as np


def dice_coeff(y_true, y_pred, smooth=1):
""" Dice coefficient used as metric"""
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    return (2. * intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)



def dice_coeff_loss(y_true, y_pred, smooth=1) :
"""Loss obtained from the die coefficient"""
    return -dice_coeff(y_true,y_pred,smooth)




def dice_coeff_generator (factor) :
"""Generates a dice coefficient with a weight factor """
    def dice_coeff(y_true, y_pred, smooth=1):
        y_true_f=K.flatten(y_true)
        y_pred_f=K.flatten(y_pred)
        intersection=K.sum(y_true_f*y_pred_f)
        return (2.*intersection+smooth)/(K.sum(y_true_f)+K.sum(y_pred_f)+smooth)

    return dice_coeff



def dice_loss_generator (factor) :
""" Generates a weighted loss from the weighted dice coefficient """
    def dice_loss(y_true, y_pred, smooth=1) :
        return 1-dice_coeff_generator(factor)(y_true,y_pred,smooth)
    return dice_loss



def sum_dice_cross_entropy (y_true, y_pred):
"""Loss function whih combines crossentropy and dice coefficient """
    return (tf.keras.losses.SparseCategoricalCrossentropy(from_logits = False)(y_true,y_pred)+1-dice_coeff(y_true,y_pred))



def dice_coef_eval(y_true, y_pred, smooth = 0.01):
""" Die coefficient used for evaluations, predicts pixels with nerves """
    y_pred = tf.cast(y_pred > 0.5, dtype=tf.float32)
    target = tf.cast(y_true > 0.5, dtype=tf.float32)
    inse = tf.reduce_sum(tf.multiply(y_pred, y_true), axis=(1,2,3))
    l = tf.reduce_sum(y_pred, axis=(1,2,3))
    r = tf.reduce_sum(y_true, axis=(1,2,3))
    hard_dice = (2. * inse + smooth) / (l + r + smooth)
    hard_dice = tf.reduce_mean(hard_dice, name='hard_dice')
    return hard_dice
