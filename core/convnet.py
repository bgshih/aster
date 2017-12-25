from abc import ABCMeta
from abc import abstractmethod

import tensorflow as tf
from tensorflow.contrib.framework import arg_scope


class Convnet(object):
  __metaclass__ = ABCMeta

  def __init__(self,
               conv_hyperparams=None,
               summarize_activations=False,
               is_training=True):
    self._conv_hyperparams = conv_hyperparams
    self._summarize_activations = summarize_activations
    self._is_training = is_training

  def preprocess(self, resized_inputs, scope=None):
    with tf.variable_scope(scope, 'ConvnetPreprocess', [resized_inputs]):
      preprocessed_inputs = (2.0 / 255.0) * resized_inputs - 1.0
      if self._summarize_activations:
        tf.summary.image('preprocessed_inputs', preprocessed_inputs, max_outputs=1)
    return preprocessed_inputs
  
  def extract_features(self, preprocessed_inputs, scope=None):
    with tf.variable_scope(scope, 'Convnet', [preprocessed_inputs]):
      shape_assert = self._shape_check(preprocessed_inputs)
      if shape_assert is None:
        shape_assert = tf.Assert(True)
      with tf.control_dependencies([shape_assert]), \
           arg_scope(self._conv_hyperparams):
        feature_maps_dict = self._extract_features(preprocessed_inputs)
        if self._summarize_activations:
          for k, tensor in feature_maps_dict.items():
            tf.summary.histogram('Activations/' + k, tensor)
      return self._output_endpoints(feature_maps_dict)

  def _shape_check(self, preprocessed_inputs):
    return None
  
  @abstractmethod
  def _output_endpoints(self, feature_maps_dict):
    raise NotImplementedError
