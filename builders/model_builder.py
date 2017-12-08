import tensorflow as tf

from rare.builders import feature_extractor_builder
from rare.builders import loss_builder
from rare.builders import label_map_builder
from rare.meta_architectures import attention_recognition_model
from rare.protos import model_pb2


def build(config, is_training):
  if not isinstance(config, model_pb2.RecognitionModel):
    raise ValueError('config not of type '
                     'model_pb2.RecognitionModel')

  model_oneof = config.WhichOneof('recognition_model_oneof')
  if model_oneof == 'attention_recognition_model':
    return _build_attention_recognition_model(config.attention_recognition_model, is_training)
  else:
    raise ValueError('Unknown recognition_model_oneof: {}'.format(model_oneof))


def _build_attention_recognition_model(model_config, is_training):
  feature_extractor_object = feature_extractor_builder.build(
    model_config.feature_extractor,
    is_training=is_training
  )
  label_map_object = label_map_builder.build(
    model_config.label_map
  )
  loss_object = loss_builder.build(
    model_config.loss
  )

  model_object = attention_recognition_model.AttentionRecognitionModel(
    feature_extractor=feature_extractor_object,
    label_map=label_map_object,
    loss=loss_object
  )
  return model_object
