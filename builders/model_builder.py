import tensorflow as tf

from aster.builders import spatial_transformer_builder
from aster.builders import feature_extractor_builder
from aster.builders import predictor_builder
from aster.builders import loss_builder
from aster.meta_architectures import multi_predictors_recognition_model
from aster.protos import model_pb2


def build(config, is_training):
  if not isinstance(config, model_pb2.Model):
    raise ValueError('config not of type model_pb2.Model')
  model_oneof = config.WhichOneof('model_oneof')
  if model_oneof == 'multi_predictors_recognition_model':
    return _build_multi_predictors_recognition_model(
      config.multi_predictors_recognition_model, is_training)
  else:
    raise ValueError('Unknown model_oneof: {}'.format(model_oneof))

def _build_multi_predictors_recognition_model(config, is_training):
  if not isinstance(config, model_pb2.MultiPredictorsRecognitionModel):
    raise ValueError('config not of type model_pb2.MultiPredictorsRecognitionModel')
  
  spatial_transformer_object = None
  if config.HasField('spatial_transformer'):
    spatial_transformer_object = spatial_transformer_builder.build(
      config.spatial_transformer, is_training)

  feature_extractor_object = feature_extractor_builder.build(
    config.feature_extractor,
    is_training=is_training
  )
  predictors_dict = {
    predictor_config.name : predictor_builder.build(predictor_config, is_training=is_training)
    for predictor_config in config.predictor
  }
  regression_loss_object = (
    None if not config.keypoint_supervision else
    loss_builder.build(config.regression_loss))
    
  model_object = multi_predictors_recognition_model.MultiPredictorsRecognitionModel(
    spatial_transformer=spatial_transformer_object,
    feature_extractor=feature_extractor_object,
    predictors_dict=predictors_dict,
    keypoint_supervision=config.keypoint_supervision,
    regression_loss=regression_loss_object,
    is_training=is_training,
  )
  return model_object
