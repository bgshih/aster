import tensorflow as tf
from tensorflow.contrib import rnn

from rare.protos import predictor_pb2
from rare.builders import rnn_cell_builder
from rare.builders import label_map_builder
from rare.builders import loss_builder
from rare.builders import hyperparams_builder
from rare.predictors import attention_predictor
# from rare.predictors import attention_predictor_with_lm


def build(config, is_training):
  if not isinstance(config, predictor_pb2.Predictor):
    raise ValueError('config not of type predictor_pb2.AttentionPredictor')
  predictor_oneof = config.WhichOneof('predictor_oneof')

  if predictor_oneof == 'attention_predictor':
    predictor_config = config.attention_predictor
    rnn_cell_object = rnn_cell_builder.build(predictor_config.rnn_cell)
    rnn_regularizer_object = hyperparams_builder._build_regularizer(predictor_config.rnn_regularizer)
    label_map_object = label_map_builder.build(predictor_config.label_map)
    loss_object = loss_builder.build(predictor_config.loss)
    if not predictor_config.HasField('lm_rnn_cell'):
      lm_rnn_cell_object = None
    else:
      lm_rnn_cell_object = _build_language_model_rnn_cell(predictor_config.lm_rnn_cell)
      
    attention_predictor_object = attention_predictor.AttentionPredictor(
      rnn_cell=rnn_cell_object,
      rnn_regularizer=rnn_regularizer_object,
      num_attention_units=predictor_config.num_attention_units,
      max_num_steps=predictor_config.max_num_steps,
      multi_attention=predictor_config.multi_attention,
      beam_width=predictor_config.beam_width,
      reverse=predictor_config.reverse,
      label_map=label_map_object,
      loss=loss_object,
      sync=predictor_config.sync,
      lm_rnn_cell=lm_rnn_cell_object,
      is_training=is_training
    )
    return attention_predictor_object
  else:
    raise ValueError('Unknown predictor_oneof: {}'.format(predictor_oneof))


def _build_language_model_rnn_cell(config):
  if not isinstance(config, predictor_pb2.LanguageModelRnnCell):
    raise ValueError('config not of type predictor_pb2.LanguageModelRnnCell')
  rnn_cell_list = [
    rnn_cell_builder.build(rnn_cell_config) for rnn_cell_config in config.rnn_cell
  ]
  lm_rnn_cell = rnn.MultiRNNCell(rnn_cell_list)
  return lm_rnn_cell
