# SVEN: https://github.com/gao-lab/SVEN

import tensorflow as tf
import tensorflow.keras as keras
from tensorflow.keras import layers

from sven.blocks import conv_block, Rconv_block, dilated_residual
from sven.layers import StochasticShift, StochasticReverseComplement, SwitchReverse

# holistic model: DNA accessibility
def acc_hol_model(input_length):
    #input sequence
    sequence = keras.Input(shape=(input_length, 4), name='sequence')
    current = sequence
    #rc sequence
    current, reverse_bool = StochasticReverseComplement()(current)
    #shift sequence
    current = StochasticShift(3)(current)
    #first conv block
    current = layers.Conv1D(
        filters=384,
        kernel_size=15,
        strides=1,
        padding="same",
        dilation_rate=1,
        kernel_initializer='he_normal',
        kernel_regularizer=keras.regularizers.l2(0)
    )(current)
    current = Rconv_block(current, filters=384)
    current = layers.MaxPool1D(pool_size=2, padding="same")(current)
    filter_num = 384
    for x in range(6):
        current = Rconv_block(current, filters=filter_num, kernel_size=5, residual=False)
        current = Rconv_block(current, filters=filter_num)
        current = layers.MaxPool1D(pool_size=2, padding="same")(current)
        filter_num = int(filter_num*1.1496)
    
    #dr block
    current = dilated_residual(current, filters=766, kernel_size=3, rate_mult=1.5, dropout=0.3, repeat=12)
    #conv block
    current = layers.Cropping1D(cropping=64)(current)
    current = conv_block(current, filters=1536, kernel_size=1,dropout=0.05)

    current = layers.Activation('gelu')(current)

    current = layers.Dense(684)(current)
    current = layers.Activation('softplus')(current)
    outputs = SwitchReverse()([current, reverse_bool])
    
    model = keras.Model(inputs=sequence, outputs=outputs, name="acc_hol_model")
    return model

# build-in model: acc
def acc_build_in_model(input_length=131_072):
    #input sequence
    sequence = keras.Input(shape=(input_length, 4), name='sequence')
    current = sequence
    #first conv block
    current = layers.Conv1D(
        filters=384,
        kernel_size=15,
        strides=1,
        padding="same",
        dilation_rate=1,
        kernel_initializer='he_normal',
        kernel_regularizer=keras.regularizers.l2(0)
    )(current)
    current = Rconv_block(current, filters=384)
    current = layers.MaxPool1D(pool_size=2, padding="same")(current)
    filter_num = 384
    for x in range(6):
        current = Rconv_block(current, filters=filter_num, kernel_size=5, residual=False)
        current = Rconv_block(current, filters=filter_num)
        current = layers.MaxPool1D(pool_size=2, padding="same")(current)
        filter_num = int(filter_num*1.1496)
    
    #dr block
    current = dilated_residual(current, filters=766, kernel_size=3, rate_mult=1.5, dropout=0.3, repeat=12)
    #conv block
    current = layers.Cropping1D(cropping=64)(current)
    current = conv_block(current, filters=1536, kernel_size=1,dropout=0.05)

    current = layers.Activation('gelu')(current)

    current = layers.Dense(684)(current)
    outputs = layers.Activation('softplus')(current)

    model = keras.Model(inputs=sequence, outputs=outputs, name="acc_hol_model")
    return model

# holistic model: Histone modification
def his_hol_model(input_length):
    #acc model
    acc_model = acc_build_in_model()
    
    #input sequence
    sequence = keras.Input(shape=(input_length, 4), name='sequence')
    current = sequence
    #rc sequence
    current, reverse_bool = StochasticReverseComplement()(current)
    #shift sequence
    current = StochasticShift(3)(current)
    acc_input = current
    acc_labels_sr = acc_model(acc_input)
    
    #first conv block
    current = layers.Conv1D(
        filters=384,
        kernel_size=15,
        strides=1,
        padding="same",
        dilation_rate=1,
        kernel_initializer='he_normal',
        kernel_regularizer=keras.regularizers.l2(0)
    )(current)
    current = Rconv_block(current, filters=384)
    current = layers.MaxPool1D(pool_size=2, padding="same")(current)
    filter_num = 384
    for x in range(6):
        current = Rconv_block(current, filters=filter_num, kernel_size=5, residual=False)
        current = Rconv_block(current, filters=filter_num)
        current = layers.MaxPool1D(pool_size=2, padding="same")(current)
        filter_num = int(filter_num*1.1496)

    #dr block
    current = dilated_residual(current, filters=766, kernel_size=3, rate_mult=1.5, dropout=0.3, repeat=12)
    #conv block
    current = layers.Cropping1D(cropping=64)(current)
    
    current = layers.Concatenate()([current, acc_labels_sr])
    current = conv_block(current, filters=1536, kernel_size=1,dropout=0.05)

    current = layers.Activation('gelu')(current)

    current = layers.Dense(1976)(current)
    current = layers.Activation('softplus')(current)
    outputs = SwitchReverse()([current, reverse_bool])
    
    model = keras.Model(inputs=sequence, outputs=outputs, name="his_hol_model")
    return model

# holistic model: TF binding
def tf_hol_model(input_length):
    #input sequence
    sequence = keras.Input(shape=(input_length, 4), name='sequence')
    current = sequence
    #rc sequence
    current, reverse_bool = StochasticReverseComplement()(current)
    #shift sequence
    current = StochasticShift(3)(current)
    #first conv block
    current = layers.Conv1D(
        filters=768,
        kernel_size=15,
        strides=1,
        padding="same",
        dilation_rate=1,
        kernel_initializer='he_normal',
        kernel_regularizer=keras.regularizers.l2(0)
    )(current)
    current = Rconv_block(current, filters=768)
    current = layers.MaxPool1D(pool_size=2)(current)
    filter_num = 768
    for x in range(6):
        current = Rconv_block(current, filters=filter_num, kernel_size=5, residual=False)
        current = Rconv_block(current, filters=filter_num)
        current = layers.MaxPool1D(pool_size=2)(current)
        filter_num = int(filter_num*1.1494)

    #dr block
    current = dilated_residual(current, filters=1536, kernel_size=3, rate_mult=1.5, dropout=0.3, repeat=11)
    #conv block
    current = layers.Cropping1D(cropping=64)(current)
    current = conv_block(current, filters=1536, kernel_size=1, dropout=0.05)

    current = layers.Activation('gelu')(current)

    current = layers.Dense(2540)(current)
    current = layers.Activation('softplus')(current)
    outputs = SwitchReverse()([current, reverse_bool])
    
    model = keras.Model(inputs=sequence, outputs=outputs, name="tf_hol_model")
    return model

# separate model: F1 model, for TF binding only
def sep_f1_model(input_length):
    #input sequence
    sequence = keras.Input(shape=(input_length, 4), name='sequence')
    current = sequence
    #rc sequence
    current, reverse_bool = StochasticReverseComplement()(current)
    #shift sequence
    current = StochasticShift(3)(current)
    #first conv block
    current = layers.Conv1D(
        filters=128,
        kernel_size=15,
        strides=1,
        padding="same",
        dilation_rate=1,
        kernel_initializer='he_normal',
        kernel_regularizer=keras.regularizers.l2(0)
    )(current)
    current = Rconv_block(current, filters=128)
    current = layers.MaxPool1D(pool_size=2, padding="same")(current)
    filter_num = 192
    for x in range(6):
        current = Rconv_block(current, filters=filter_num, kernel_size=5, residual=False)
        current = Rconv_block(current, filters=filter_num)
        current = layers.MaxPool1D(pool_size=2, padding="same")(current)
        filter_num = int(filter_num*1.15)
    
    #dr block
    current = dilated_residual(current, filters=384, kernel_size=3, rate_mult=1.5, dropout=0.3, repeat=12)
    #conv block
    current = layers.Cropping1D(cropping=64)(current)
    current = conv_block(current, filters=512, kernel_size=1, dropout=0.05)

    current = layers.Activation('gelu')(current)

    current = layers.Dense(1)(current)
    current = layers.Activation('softplus')(current)
    outputs = SwitchReverse()([current, reverse_bool])
    
    model = keras.Model(inputs=sequence, outputs=outputs, name="tf_sep_f1_model")
    return model

# separate model: F2 model, for TF binding and histone modification
def sep_f2_model(input_length):
    #acc model
    acc_model = acc_build_in_model()
    
    #input sequence
    sequence = keras.Input(shape=(input_length, 4), name='sequence')
    current = sequence
    #rc sequence
    current, reverse_bool = StochasticReverseComplement()(current)
    #shift sequence
    current = StochasticShift(3)(current)
    acc_input = current
    #first conv block
    current = layers.Conv1D(
        filters=192,
        kernel_size=15,
        strides=1,
        padding="same",
        dilation_rate=1,
        kernel_initializer='he_normal',
        kernel_regularizer=keras.regularizers.l2(0)
    )(current)
    current = Rconv_block(current, filters=192)
    current = layers.MaxPool1D(pool_size=2, padding="same")(current)
    filter_num = 192
    for x in range(6):
        current = Rconv_block(current, filters=filter_num, kernel_size=5, residual=False)
        current = Rconv_block(current, filters=filter_num)
        current = layers.MaxPool1D(pool_size=2, padding="same")(current)
        filter_num = int(filter_num*1.1496)

    #dr block
    current = dilated_residual(current, filters=384, kernel_size=3, rate_mult=1.5, dropout=0.3, repeat=12)
    #conv block
    current = layers.Cropping1D(cropping=64)(current)
    acc_labels_sr = acc_model(acc_input)
    current = layers.Concatenate()([current, acc_labels_sr])
    current = conv_block(current, filters=512, kernel_size=1, dropout=0.05)

    current = layers.Activation('gelu')(current)

    current = layers.Dense(1)(current)
    current = layers.Activation('softplus')(current)
    outputs = SwitchReverse()([current, reverse_bool])
    
    model = keras.Model(inputs=sequence, outputs=outputs, name="tf_his_sep_model")
    return model