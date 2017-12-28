import tensorflow as tf

from rare.protos import predictor_pb2
from rare.builders import rnn_cell_builder
from rare.builders import label_map_builder
from rare.builders import loss_builder
from rare.builders import hyperparams_builder
from rare.predictors import attention_predictor


def build(config, is_training):
  if not isinstance(config, predictor_pb2.Predictor):
    raise ValueError('config not of type predictor_pb2.AttentionPredictor')
  predictor_oneof = config.WhichOneof('predictor_oneof')

  if predictor_oneof == 'bahdanau_attention_predictor':
    predictor_config = config.bahdanau_attention_predictor
    rnn_cell_object = rnn_cell_builder.build(predictor_config.rnn_cell)
    rnn_regularizer_object = hyperparams_builder._build_regularizer(predictor_config.rnn_regularizer)
    label_map_object = label_map_builder.build(predictor_config.label_map)
    loss_object = loss_builder.build(predictor_config.loss)
    attention_predictor_object = attention_predictor.BahdanauAttentionPredictor(
      rnn_cell=rnn_cell_object,
      rnn_regularizer=rnn_regularizer_object,
      num_attention_units=predictor_config.num_attention_units,
      max_num_steps=predictor_config.max_num_steps,
      multi_attention=predictor_config.multi_attention,
      beam_width=predictor_config.beam_width,
      reverse=predictor_config.reverse,
      label_map=label_map_object,
      loss=loss_object,
      is_training=is_training
    )
    return attention_predictor_object
  else:
    raise ValueError('Unknown predictor_oneof: {}'.format(predictor_oneof))
