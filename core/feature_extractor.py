from abc import ABCMeta
from abc import abstractmethod
import functools

import tensorflow as tf
from tensorflow.contrib.layers import conv2d, max_pool2d
from tensorflow.contrib.framework import arg_scope

from rare.core import feature_extractor_pb2
from rare.core import bidirectional_rnn
from rare.utils import visualization_utils, shape_utils


class FeatureExtractor(object):
  """Abstract class for feature extractor."""
  __metaclass__ = ABCMeta

  def __init__(self,
               summarize_inputs=False,
               brnn_fn_list=[],
               is_training=True):
    self._summarize_inputs = summarize_inputs
    self._brnn_fn_list = brnn_fn_list
    self._is_training = is_training

  def preprocess(self, resized_inputs, scope=None):
    with tf.variable_scope(scope, 'FeatureExtractorPreprocess', [resized_inputs]):
      preprocessed_inputs = (2.0 / 255.0) * resized_inputs - 1.0
    if self._summarize_inputs:
      tf.summary.image('preprocessed_inputs', preprocessed_inputs, max_outputs=1)
    return preprocessed_inputs

  def extract_features(self, preprocessed_inputs, scope=None):
    with tf.variable_scope(scope, 'FeatureExtractor', [preprocessed_inputs]):
      feature_maps = self._extract_features(preprocessed_inputs)

    if len(self._brnn_fn_list) > 0:
      feature_sequences_list = []

      for i, feature_map in enumerate(feature_maps):
        shape_assert = tf.Assert(
          tf.equal(tf.shape(feature_map)[1], 1),
          ['Feature map height must be 1 if bidirectional RNN is going to be applied.']
        )
        batch_size, _, _, map_depth = shape_utils.combined_static_and_dynamic_shape(feature_map)
        with tf.control_dependencies([shape_assert]):
          feature_sequence = tf.reshape(feature_map, [batch_size, -1, map_depth])
        for j, brnn_fn in enumerate(self._brnn_fn_list):
          brnn_object = brnn_fn()
          feature_sequence = brnn_object.predict(feature_sequence, scope='BidirectionalRnn_Branch_{}_{}'.format(i, j))
        feature_sequences_list.append(feature_sequence)

      feature_maps = tf.expand_dims(feature_sequences_list, axis=1)

    return feature_maps

  @abstractmethod
  def _extract_features(self, preprocessed_inputs):
    pass


def build(config, is_training):
  if not isinstance(config, feature_extractor_pb2.FeatureExtractor):
    raise ValueError('config not of type '
                     'feature_extractor_pb2.FeatureExtractor')
  feature_extractor_oneof = config.WhichOneof('feature_extractor_oneof')
  if feature_extractor_oneof == 'baseline_feature_extractor':
    from rare.feature_extractors import baseline_feature_extractor
    return baseline_feature_extractor.build(
      config.baseline_feature_extractor,
      is_training
    )
  elif feature_extractor_oneof == 'resnet_feature_extractor':
    from rare.feature_extractors import resnet_feature_extractor
    return resnet_feature_extractor.build(config.resnet_feature_extractor, is_training)
  else:
    raise ValueError('Unknown feature_extractor_oneof: {}'.format(feature_extractor_oneof))

