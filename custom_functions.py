import glob
from natsort import natsorted
import numpy as np

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

from tensorflow.keras import backend as K


'''function for making dataset'''
def file_to_records(filename):
    return tf.data.TFRecordDataset(filename, compression_type='ZLIB')

class SeqDataset:
    def __init__(self,data_dir,batch_size,seq_len=131_072,data_mode="test",tfr_pattern=None):
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.data_mode = data_mode
        self.seq_len = seq_len
        self.tfr_pattern=tfr_pattern
        #make dataset
        self.make_dataset()
    
    #decode data
    def generate_parser(self):
        def parse_proto(example_protos):
            features={
                'sequence':tf.io.FixedLenFeature([], tf.string),
                'target':tf.io.FixedLenFeature([], tf.string)
            }
            parsed_features = tf.io.parse_single_example(example_protos, features=features)
            #decode sequence and reshape
            sequence = tf.io.decode_raw(parsed_features['sequence'], tf.uint8)
            sequence = tf.reshape(sequence, [self.seq_len, 4])
            sequence = tf.cast(sequence, tf.float32)
            #return data
            return sequence
        return parse_proto

    def make_dataset(self):
        #data path
        if self.tfr_pattern==None:
            self.tfr_pattern="-0-*.tfr"
        tfr_path=self.data_dir+self.data_mode+self.tfr_pattern
        tfr_files=natsorted(glob.glob(tfr_path))

        dataset = tf.data.Dataset.from_tensor_slices(tfr_files)
        if self.data_mode=='train':
            dataset = dataset.repeat()
            dataset = dataset.interleave(file_to_records,cycle_length=6,num_parallel_calls=tf.data.AUTOTUNE)
            #shuffle
            dataset = dataset.shuffle(buffer_size=512,reshuffle_each_iteration=True)
        else:
            dataset = dataset.flat_map(file_to_records)
        
        dataset = dataset.map(self.generate_parser())
        dataset = dataset.batch(self.batch_size)  
        dataset = dataset.prefetch(tf.data.AUTOTUNE)
        #hold
        self.dataset = dataset


'''function for shifting a sequence left or right by shift_amount.'''
def shift_sequence(seq, shift, pad_value=0.25): 
    if seq.shape.ndims != 3:
        raise ValueError('input sequence should be rank 3')
    input_shape = seq.shape

    pad = pad_value * tf.ones_like(seq[:, 0:tf.abs(shift), :])

    def _shift_right(_seq):
        # shift is positive
        sliced_seq = _seq[:, :-shift:, :]
        return tf.concat([pad, sliced_seq], axis=1)

    def _shift_left(_seq):
        # shift is negative
        sliced_seq = _seq[:, -shift:, :]
        return tf.concat([sliced_seq, pad], axis=1)

    sseq = tf.cond(tf.greater(shift, 0),
                    lambda: _shift_right(seq),
                    lambda: _shift_left(seq))
    sseq.set_shape(input_shape)

    return sseq


#custom layers
"""Stochastically shift a one hot encoded DNA sequence."""
class StochasticShift(layers.Layer):
    def __init__(self, shift_max=0, symmetric=True, pad='uniform'):
        super(StochasticShift, self).__init__()
        self.shift_max = shift_max
        self.symmetric = symmetric
        if self.symmetric:
            self.augment_shifts = tf.range(-self.shift_max, self.shift_max+1)
        else:
            self.augment_shifts = tf.range(0, self.shift_max+1)
        self.pad = pad
    
    def call(self, seq_1hot, training=None):
        if training:
            shift_i = tf.random.uniform(shape=[], minval=0, dtype=tf.int64,
                                        maxval=len(self.augment_shifts))
            shift = tf.gather(self.augment_shifts, shift_i)
            sseq_1hot = tf.cond(tf.not_equal(shift, 0),
                                lambda: shift_sequence(seq_1hot, shift),
                                lambda: seq_1hot)
            return sseq_1hot
        else:
            return seq_1hot
    
    def get_config(self):
        config = super().get_config().copy()
        config.update({
            'shift_max': self.shift_max,
            'symmetric': self.symmetric,
            'pad': self.pad
            })
        return config


"""Stochastically reverse complement a one hot encoded DNA sequence."""
class StochasticReverseComplement(layers.Layer):
    def __init__(self):
        super(StochasticReverseComplement, self).__init__()
    
    def call(self, seq_1hot, training=None):
        if training:
            rc_seq_1hot = tf.gather(seq_1hot, [3, 2, 1, 0], axis=-1)
            rc_seq_1hot = tf.reverse(rc_seq_1hot, axis=[1])
            reverse_bool = tf.random.uniform(shape=[]) > 0.5
            src_seq_1hot = tf.cond(reverse_bool, lambda: rc_seq_1hot, lambda: seq_1hot)
            return src_seq_1hot, reverse_bool
        else:
            return seq_1hot, tf.constant(False)


"""Reverse predictions if the inputs were reverse complemented."""
class SwitchReverse(layers.Layer): 
    def __init__(self):
        super(SwitchReverse, self).__init__()
    
    def call(self, x_reverse):
        x = x_reverse[0]
        reverse = x_reverse[1]

        return tf.cond(reverse, lambda: tf.reverse(x, axis=[1]), lambda: x)


"""Pooling operation with optional weights."""
class SoftmaxPool1D(layers.Layer):
    def __init__(self,
               pool_size: int = 2,
               per_channel: bool = True,
               init_gain: float = 2.0):

        super(SoftmaxPool1D, self).__init__()
        self.pool_size = pool_size
        self.per_channel = per_channel
        self.init_gain = init_gain
        self.logit_linear = None

    def build(self, input_shape):
        self.num_channels = input_shape[-1]
        self.logit_linear = layers.Dense(
            units=self.num_channels if self.per_channel else 1,
            use_bias=False,
            kernel_initializer=keras.initializers.Identity(self.init_gain))

    def call(self, inputs):
        _, seq_length, num_channels = inputs.shape
        inputs = tf.reshape(inputs,
            (-1, seq_length // self.pool_size, self.pool_size, num_channels))
        return tf.reduce_sum(
            inputs * tf.nn.softmax(self.logit_linear(inputs), axis=-2),
            axis=-2)

    def get_config(self):
        config = super().get_config().copy()
        config.update({
        'pool_size': self.pool_size,
        'init_gain': self.init_gain
        })
        return config

#metrics
class PearsonR(keras.metrics.Metric):
    def __init__(self, num_targets, summarize=True, name='pearsonr', **kwargs):
        super(PearsonR, self).__init__(name=name, **kwargs)
        self._summarize = summarize
        self._shape = (num_targets,)
        self._count = self.add_weight(name='count', shape=self._shape, initializer='zeros')

        self._product = self.add_weight(name='product', shape=self._shape, initializer='zeros')
        self._true_sum = self.add_weight(name='true_sum', shape=self._shape, initializer='zeros')
        self._true_sumsq = self.add_weight(name='true_sumsq', shape=self._shape, initializer='zeros')
        self._pred_sum = self.add_weight(name='pred_sum', shape=self._shape, initializer='zeros')
        self._pred_sumsq = self.add_weight(name='pred_sumsq', shape=self._shape, initializer='zeros')

    def update_state(self, y_true, y_pred, sample_weight=None):
        y_true = tf.cast(y_true, 'float32')
        y_pred = tf.cast(y_pred, 'float32')

        if len(y_true.shape) == 2:
            reduce_axes = 0
        else:
            reduce_axes = [0,1]

        product = tf.reduce_sum(tf.multiply(y_true, y_pred), axis=reduce_axes)
        self._product.assign_add(product)

        true_sum = tf.reduce_sum(y_true, axis=reduce_axes)
        self._true_sum.assign_add(true_sum)

        true_sumsq = tf.reduce_sum(tf.math.square(y_true), axis=reduce_axes)
        self._true_sumsq.assign_add(true_sumsq)

        pred_sum = tf.reduce_sum(y_pred, axis=reduce_axes)
        self._pred_sum.assign_add(pred_sum)

        pred_sumsq = tf.reduce_sum(tf.math.square(y_pred), axis=reduce_axes)
        self._pred_sumsq.assign_add(pred_sumsq)

        count = tf.ones_like(y_true)
        count = tf.reduce_sum(count, axis=reduce_axes)
        self._count.assign_add(count)

    def result(self):
        true_mean = tf.divide(self._true_sum, self._count)
        true_mean2 = tf.math.square(true_mean)
        pred_mean = tf.divide(self._pred_sum, self._count)
        pred_mean2 = tf.math.square(pred_mean)

        term1 = self._product
        term2 = -tf.multiply(true_mean, self._pred_sum)
        term3 = -tf.multiply(pred_mean, self._true_sum)
        term4 = tf.multiply(self._count, tf.multiply(true_mean, pred_mean))
        covariance = term1 + term2 + term3 + term4

        true_var = self._true_sumsq - tf.multiply(self._count, true_mean2)
        pred_var = self._pred_sumsq - tf.multiply(self._count, pred_mean2)
        pred_var = tf.where(tf.greater(pred_var, 1e-12),
                            pred_var,
                            np.inf*tf.ones_like(pred_var))
        
        tp_var = tf.multiply(tf.math.sqrt(true_var), tf.math.sqrt(pred_var))
        correlation = tf.divide(covariance, tp_var)

        if self._summarize:
            return tf.reduce_mean(correlation)
        else:
            return correlation

    def reset_state(self):
        K.batch_set_value([(v, np.zeros(self._shape)) for v in self.variables])