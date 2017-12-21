import functools

import tensorflow as tf

from rare.core import rnn_cell_pb2
from rare.core import hyperparams


def _hook_rnn_call_fn(call_fn):
  """Hook a RNN.__call__ function so that it applies reguarlizer to weights after calling."""


def build(rnn_cell_config):
  if not isinstance(rnn_cell_config, rnn_cell_pb2.RnnCell):
    raise ValueError('rnn_cell_config not of type '
                     'rnn_cell_pb2.RnnCell')
  rnn_cell_oneof = rnn_cell_config.WhichOneof('rnn_cell_oneof')

  if rnn_cell_oneof == 'lstm_cell':
    lstm_cell_config = rnn_cell_config.lstm_cell
    weights_initializer_object = hyperparams._build_initializer(
      lstm_cell_config.initializer)
    lstm_cell_object = tf.contrib.rnn.LSTMCell(
      lstm_cell_config.num_units,
      use_peepholes=lstm_cell_config.use_peepholes,
      forget_bias=lstm_cell_config.forget_bias,
      initializer=weights_initializer_object
    )
    return lstm_cell_object

  elif rnn_cell_oneof == 'gru_cell':
    gru_cell_config = rnn_cell_config.gru_cell
    weights_initializer_object = hyperparams._build_initializer(
      gru_cell_config.initializer)
    gru_cell_object = tf.contrib.rnn.GRUCell(
      gru_cell_config.num_units,
      kernel_initializer=weights_initializer_object
    )
    return gru_cell_object

  else:
    raise ValueError('Unknown rnn_cell_oneof: {}'.format(rnn_cell_oneof))
