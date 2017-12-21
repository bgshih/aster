from abc import ABCMeta
from abc import abstractmethod

import tensorflow as tf

from rare.models import model_pb2


class Model(object):
  __metaclass__ = ABCMeta

  def __init__(self, feature_extractor, is_training):
    self._feature_extractor = feature_extractor
    self._is_training = is_training
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


def build(model_config, is_training):
  if not isinstance(model_config, model_pb2.Model):
    raise ValueError('model_config not of type '
                     'model_pb2.Model')

  model_oneof = model_config.WhichOneof('model_oneof')
  if model_oneof == 'attention_recognition_model':
    from rare.models import attention_recognition_model
    return attention_recognition_model.build(model_config.attention_recognition_model, is_training)
  elif model_oneof == 'ctc_recognition_model':
    from rare.models import ctc_recognition_model
    return ctc_recognition_model.build(model_config.ctc_recognition_model, is_training)
  else:
    raise ValueError('Unknown model_oneof: {}'.format(model_oneof))
