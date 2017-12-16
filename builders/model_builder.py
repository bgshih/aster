import tensorflow as tf

from rare.builders import feature_extractor_builder
from rare.builders import predictor_builder
from rare.builders import loss_builder
from rare.builders import rnn_cell_builder
from rare.builders import label_map_builder
from rare.meta_architectures import attention_recognition_model
from rare.meta_architectures import ctc_recognition_model
from rare.protos import model_pb2


def build(model_config, is_training):
  if not isinstance(model_config, model_pb2.RecognitionModel):
    raise ValueError('model_config not of type '
                     'model_pb2.RecognitionModel')

  model_oneof = model_config.WhichOneof('recognition_model_oneof')
  if model_oneof == 'attention_recognition_model':
    return _build_attention_recognition_model(model_config.attention_recognition_model, is_training)
  elif model_oneof == 'ctc_recognition_model':
    return _build_ctc_recognition_model(model_config.ctc_recognition_model, is_training)
  else:
    raise ValueError('Unknown recognition_model_oneof: {}'.format(model_oneof))

def _build_attention_recognition_model(model_config, is_training):
  feature_extractor_object = feature_extractor_builder.build(
    model_config.feature_extractor,
    is_training=is_training
  )
  predictor_object = predictor_builder.build(model_config.predictor, is_training=is_training)
  label_map_object = label_map_builder.build(model_config.label_map)
  loss_object = loss_builder.build(model_config.loss)

  model_object = attention_recognition_model.AttentionRecognitionModel(
    feature_extractor=feature_extractor_object,
    predictor=predictor_object,
    label_map=label_map_object,
    loss=loss_object,
    is_training=is_training
  )
  return model_object

def _build_ctc_recognition_model(model_config, is_training):
  feature_extractor_object = feature_extractor_builder.build(
    model_config.feature_extractor,
    is_training=is_training
  )
  label_map_object = label_map_builder.build(model_config.label_map)

  bidirectional_rnn_cells_list = [
    rnn_cell_builder.build(rnn_cell_config) for rnn_cell_config in model_config.bidirectional_rnn_cell
  ]

  model_object = ctc_recognition_model.CtcRecognitionModel(
    feature_extractor=feature_extractor_object,
    bidirectional_rnn_cell_list=bidirectional_rnn_cells_list,
    label_map=label_map_object,
    loss=None,
    is_training=is_training)
  return model_object
