import tensorflow as tf

from rare.predictors import attention_predictor
from rare.protos import predictor_pb2
from rare.builders import rnn_cell_builder


def build(config, is_training):
  if not isinstance(config, predictor_pb2.Predictor):
    raise ValueError('config not of type predictor_pb2.predictor')
  predictor_oneof = config.WhichOneof('predictor_oneof')

  if predictor_oneof == 'bahdanau_attention_predictor':
    predictor_config = config.bahdanau_attention_predictor

    rnn_cell_object = rnn_cell_builder.build(predictor_config.rnn_cell)
    predictor_object = attention_predictor.BahdanauAttentionPredictor(
      rnn_cell=rnn_cell_object,
      num_attention_units=predictor_config.num_attention_units,
      max_num_steps=predictor_config.max_num_steps,
      is_training=is_training
    )
    return predictor_object

  else:
    raise ValueError('Unknown predictor_oneof: {}'.format(predictor_oneof))
