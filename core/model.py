from abc import ABCMeta
from abc import abstractmethod

import tensorflow as tf


class Model(object):
  __metaclass__ = ABCMeta

  def __init__(self,
               feature_extractor=None,
               is_training=True):
    self._feature_extractor = feature_extractor
    self._is_training = is_training
    self._predictors = {}
    self._groundtruth_dict = {}

  def preprocess(self, resized_inputs, scope=None):
    with tf.variable_scope(scope, 'ModelPreprocess', [resized_inputs]) as preprocess_scope:
      if resized_inputs.dtype is not tf.float32:
        raise ValueError('`preprocess` expects a tf.float32 tensor')
      preprocess_inputs = self._feature_extractor.preprocess(resized_inputs, scope=preprocess_scope)
    return preprocess_inputs

  @abstractmethod
  def predict(self, preprocessed_inputs, scope=None):
    pass

  @abstractmethod
  def loss(self, predictions_dict, scope=None):
    pass

  @abstractmethod
  def postprocess(self, predictions_dict, scope=None):
    pass

  @abstractmethod
  def provide_groundtruth(self, groundtruth_lists, scope=None):
    pass
