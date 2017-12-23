import tensorflow as tf
from tensorflow.contrib.framework import arg_scope
from tensorflow.contrib.layers import fully_connected

from rare.core import bidirectional_rnn_pb2
from rare.core import rnn_cell
from rare.core import hyperparams, hyperparams_pb2


class BidirectionalRnn(object):

  def __init__(self, fw_cell, bw_cell,
               rnn_regularizer=None, num_output_units=None, fc_hyperparams=None,
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


def build(config, is_training):
  if not isinstance(config, bidirectional_rnn_pb2.BidirectionalRnn):
    raise ValueError('config not of type bidirectional_rnn_pb2.BidirectionalRnn')

  fw_cell_object = rnn_cell.build(config.fw_bw_rnn_cell)
  bw_cell_object = rnn_cell.build(config.fw_bw_rnn_cell)
  rnn_regularizer_object = hyperparams._build_regularizer(config.rnn_regularizer)
  fc_hyperparams_object = None
  if config.num_output_units:
    if config.fc_hyperparams.op != hyperparams_pb2.Hyperparams.FC:
      raise ValueError('op type must be FC')
    fc_hyperparams_object = hyperparams.build(config.fc_hyperparams, is_training)

  return BidirectionalRnn(
    fw_cell_object, bw_cell_object,
    rnn_regularizer=rnn_regularizer_object,
    num_output_units=config.num_output_units,
    fc_hyperparams=fc_hyperparams_object,
    summarize_activations=config.summarize_activations)
