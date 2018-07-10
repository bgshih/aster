import functools

import tensorflow as tf

from aster.core import feature_extractor
from aster.protos import feature_extractor_pb2
from aster.builders import convnet_builder
from aster.builders import bidirectional_rnn_builder
from aster.builders import hyperparams_builder


def build(config, is_training):
  if not isinstance(config, feature_extractor_pb2.FeatureExtractor):
    raise ValueError('config not of type feature_extractor_pb2.FeatureExtractor')

  convnet_object = convnet_builder.build(config.convnet, is_training)
  brnn_fn_list = [
    functools.partial(bidirectional_rnn_builder.build, brnn_config, is_training)
    for brnn_config in config.bidirectional_rnn
  ]
  feature_extractor_object = feature_extractor.FeatureExtractor(
    convnet=convnet_object,
    brnn_fn_list=brnn_fn_list,
    summarize_activations=config.summarize_activations,
    is_training=is_training
  )
  return feature_extractor_object
