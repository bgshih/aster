import tensorflow as tf
from rare.protos import rnn_cell_pb2
from rare.builders import hyperparams_builder


def build(rnn_cell_config):
  if not isinstance(rnn_cell_config, rnn_cell_pb2.RnnCell):
    raise ValueError('rnn_cell_config not of type '
                     'rnn_cell_pb2.RnnCell')
  rnn_cell_oneof = rnn_cell_config.WhichOneof('rnn_cell_oneof')

  if rnn_cell_oneof == 'lstm_cell':
    lstm_cell_config = rnn_cell_config.lstm_cell

    initializer_object = hyperparams_builder._build_initializer(lstm_cell_config.initializer)
    weight_regularizer_object = hyperparams_builder._build_regularizer(lstm_cell_config.weight_regularizer)
    lstm_cell_object = tf.contrib.rnn.LSTMCell(
      lstm_cell_config.num_units,
      use_peepholes=lstm_cell_config.use_peepholes,
      forget_bias=lstm_cell_config.forget_bias,
      initializer=initializer_object
    )
    assert not hasattr(lstm_cell_object, 'weight_regularizer')
    setattr(lstm_cell_object, 'weight_regularizer', weight_regularizer_object)
    return lstm_cell_object

  elif rnn_cell_oneof == 'gru_cell':
    gru_cell_config = rnn_cell_config.gru_cell
    
    gru_cell_object = tf.contrib.rnn.GRUCell(
      gru_cell_config.num_units,
    )
    return gru_cell_object

  else:
    raise ValueError('Unknown rnn_cell_oneof: {}'.format(rnn_cell_oneof))
