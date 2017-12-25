import tensorflow as tf
from tensorflow.contrib.framework import arg_scope
from tensorflow.contrib.layers import fully_connected


class BidirectionalRnn(object):

  def __init__(self,
               fw_cell,
               bw_cell,
               rnn_regularizer=None,
               num_output_units=None,
               fc_hyperparams=None,
               summarize_activations=False):
    self._fw_cell = fw_cell
    self._bw_cell = bw_cell
    self._rnn_regularizer = rnn_regularizer
    self._num_output_units = num_output_units
    self._fc_hyperparams = fc_hyperparams
    self._summarize_activations = summarize_activations

  def predict(self, inputs, scope=None):
    with tf.variable_scope(scope, 'BidirectionalRnn', [inputs]) as scope:
      (output_fw, output_bw), _ = tf.nn.bidirectional_dynamic_rnn(
        self._fw_cell, self._bw_cell, inputs, time_major=False, dtype=tf.float32)
      rnn_outputs = tf.concat([output_fw, output_bw], axis=2)

      filter_weights = lambda vars : [x for x in vars if x.op.name.endswith('kernel')]
      tf.contrib.layers.apply_regularization(self._rnn_regularizer, filter_weights(self._fw_cell.trainable_weights))
      tf.contrib.layers.apply_regularization(self._rnn_regularizer, filter_weights(self._bw_cell.trainable_weights))

      if self._num_output_units:
        with arg_scope(self._fc_hyperparams):
          rnn_outputs = fully_connected(rnn_outputs, self._num_output_units, activation_fn=tf.nn.relu)

    if self._summarize_activations:
      max_time = rnn_outputs.get_shape()[1].value
      for t in range(max_time):
        activation_t = rnn_outputs[:,t,:]
        tf.summary.histogram('Activations/{}/Step_{}'.format(scope.name, t), activation_t)

    return rnn_outputs
