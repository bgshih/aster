from abc import ABCMeta
from abc import abstractmethod

import tensorflow as tf


class Predictor(object):
  __metaclass__ = ABCMeta

  def __init__(self, is_training):
    self._is_training = is_training

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
