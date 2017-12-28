from abc import ABCMeta
from abc import abstractmethod

import tensorflow as tf


class Predictor(object):
  __metaclass__ = ABCMeta

  def __init__(self, is_training=True):
    self._is_training = is_training
    self._groundtruth_dict = {}

  @property
  def name(self):
    return self._name

  @abstractmethod
  def predict(self, feature_maps, scope=None):
    pass

  @abstractmethod
  def loss(self, predictions_dict, scope=None):
    pass

  @abstractmethod
  def provide_groundtruth(self, groundtruth_lists, scope=None):
    pass

  @abstractmethod
  def postprocess(self, predictions_dict, scope=None):
    return predictions_dict
