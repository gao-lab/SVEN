# SVEN: https://github.com/gao-lab/SVEN
# Thanks to Calico for the Basenji codebase, which was used as a reference for this code.

import tensorflow as tf
from tensorflow.keras import layers


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