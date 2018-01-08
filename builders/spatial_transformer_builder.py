import tensorflow as tf

from rare.core import spatial_transformer
from rare.protos import spatial_transformer_pb2
from rare.builders import hyperparams_builder
from rare.builders import convnet_builder


def build(config, is_training):
  if not isinstance(config, spatial_transformer_pb2.SpatialTransformer):
    raise ValueError('config not of type spatial_transformer_pb2.SpatialTransformer')
  
  convnet_object = convnet_builder.build(config.convnet, is_training)
  fc_hyperparams_object = hyperparams_builder.build(config.fc_hyperparams, is_training)
  return spatial_transformer.SpatialTransformer(
    convnet=convnet_object,
    fc_hyperparams=fc_hyperparams_object,
    localization_image_size=(config.localization_h, config.localization_w),
    output_image_size=(config.output_h, config.output_w),
    num_control_points=config.num_control_points,
    margin=config.margin
  )
