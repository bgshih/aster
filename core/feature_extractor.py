import functools

import tensorflow as tf
from tensorflow.contrib.layers import conv2d, max_pool2d
from tensorflow.contrib.framework import arg_scope

from aster.utils import shape_utils


class FeatureExtractor(object):
  def __init__(self,
               convnet=None,
               brnn_fn_list=[],
               summarize_activations=False,
               is_training=True):
    self._convnet = convnet
    self._brnn_fn_list = brnn_fn_list
    self._summarize_activations = summarize_activations
    self._is_training = is_training

  def preprocess(self, resized_inputs, scope=None):
    with tf.variable_scope(scope, 'FeatureExtractorPreprocess', [resized_inputs]) as preproc_scope:
      preprocessed_inputs = self._convnet.preprocess(resized_inputs, preproc_scope)
    return preprocessed_inputs

  def extract_features(self, preprocessed_inputs, scope=None):
    with tf.variable_scope(scope, 'FeatureExtractor', [preprocessed_inputs]):
      feature_maps = self._convnet.extract_features(preprocessed_inputs)

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

      feature_maps = [tf.expand_dims(fmap, axis=1) for fmap in feature_sequences_list]
    return feature_maps
