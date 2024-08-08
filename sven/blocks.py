# SVEN: https://github.com/gao-lab/SVEN
# Thanks to Calico for the Basenji codebase, which was used as a reference for this code.

import tensorflow as tf
import tensorflow.keras as keras
from tensorflow.keras import layers
import numpy as np

# convolution block
def conv_block(inputs, filters=128, kernel_size=1, strides=1, 
    dilation_rate=1, l2_scale=0, dropout=0, pool_size=1, bn_momentum=0.9,
    residual=False, kernel_initializer='he_normal', padding='same',norm_gamma=None):
    
    current = inputs
    #activation
    current = layers.Activation('gelu')(current)
    #convolution
    current = layers.Conv1D(
        filters=filters,
        kernel_size=kernel_size,
        strides=strides,
        padding=padding,
        dilation_rate=dilation_rate,
        kernel_initializer=kernel_initializer,
        kernel_regularizer=keras.regularizers.l2(l2_scale)
    )(current)
    #batch normalization
    current = layers.BatchNormalization(momentum=bn_momentum, gamma_initializer=norm_gamma)(current)
    # dropout
    if dropout > 0:
        current = layers.Dropout(dropout)(current)
    # residual add
    if residual:
        current = layers.Add()([inputs,current])
    # Pool
    if pool_size > 1:
        current = layers.MaxPool1D(pool_size=pool_size,padding=padding)(current)
    #return
    return current

# residual convolution block
def Rconv_block(inputs, filters, kernel_size=1, residual=True):
    current = inputs
    current = layers.BatchNormalization(momentum=0.9)(current)
    current = layers.Activation('gelu')(current)
    current = layers.Conv1D(
        filters=filters,
        kernel_size=kernel_size,
        strides=1,
        padding="same",
        dilation_rate=1,
        kernel_initializer='he_normal',
        kernel_regularizer=keras.regularizers.l2(0)
    )(current)
    if residual:
        current = layers.Add()([inputs,current])
    return current

# dilated residual convolution block
def dilated_residual(inputs, filters, kernel_size=3, rate_mult=2, dropout=0, repeat=1, **kwargs):
    current = inputs
    # initialize dilation rate
    dilation_rate = 1

    for ri in range(repeat):
        rep_input = current
        # dilate
        current = conv_block(current,
            filters=filters,
            kernel_size=kernel_size,
            dilation_rate=dilation_rate,
            norm_gamma='ones',
            **kwargs)
        # return
        current = conv_block(current,
            filters=rep_input.shape[-1],
            dropout=dropout,
            norm_gamma='zeros',
            **kwargs)
        # residual add
        current = layers.Add()([rep_input,current])
        #update dilation rate
        dilation_rate = int(np.round(float(dilation_rate)*rate_mult))
    #current
    return current