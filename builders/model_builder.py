import tensorflow as tf

from rare.builders import feature_extractor_builder
from rare.builders import loss_builder
from rare.builders import hyperparams_builder

from rare.meta_architectures import attention_recognition_model
from rare.meta_architectures import ctc_recognition_model
from rare.protos import model_pb2


def build(config, is_training):
  if not isinstance(config, model_pb2.Model):
    raise ValueError('config not of type '
                     'model_pb2.Model')
  model_oneof = config.WhichOneof('model_oneof')
  if model_oneof == 'attention_recognition_model':
    return _build_attention_recognition_model(config.attention_recognition_model, is_training)
  elif model_oneof == 'ctc_recognition_model':
    return _build_ctc_recognition_model(config.ctc_recognition_model, is_training)
  else:
    raise ValueError('Unknown model_oneof: {}'.format(model_oneof))


def _build_attention_recognition_model(config, is_training):
  if not isinstance(config, model_pb2.AttentionRecognitionModel):
    raise ValueError('config not of type model_pb2.AttentionRecognitionModel')
  feature_extractor_object = feature_extractor_builder.build(
    config.feature_extractor,
    is_training=is_training
  )
  predictor_object = _build_attention_predictor(
    config.attention_predictor,
    is_training=is_training)
  label_map_object = label_map_builder.build(config.label_map)
  loss_object = loss_builder.build(config.loss)

  model_object = attention_recognition_model.AttentionRecognitionModel(
    feature_extractor=feature_extractor_object,
    predictor=predictor_object,
    label_map=label_map_object,
    loss=loss_object,
    is_training=is_training
  )
  return model_object

def _build_ctc_recognition_model(config, is_training):
  if not isinstance(config, model_pb2.CtcRecognitionModel):
    raise ValueError('config not of type model_pb2.CtcRecognitionModel')
  feature_extractor_object = feature_extractor_builder.build(
    config.feature_extractor,
    is_training=is_training
  )
  label_map_object = label_map_builder.build(config.label_map)
  fc_hyperparams_object = hyperparams_builder.build(
    config.fc_hyperparams,
    is_training)
  model_object = ctc_recognition_model.CtcRecognitionModel(
    feature_extractor=feature_extractor_object,
    fc_hyperparams=fc_hyperparams_object,
    label_map=label_map_object,
    is_training=is_training)
  return model_object
