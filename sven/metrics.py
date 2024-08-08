# SVEN: https://github.com/gao-lab/SVEN
# Thanks to Calico for the Basenji codebase, which was used as a reference for this code.

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import backend as K

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