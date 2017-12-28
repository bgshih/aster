import logging
import functools

import tensorflow as tf
from tensorflow.contrib import seq2seq

from rare.core import model
from rare.utils import shape_utils


class MultiPredictorsRecognitionModel(model.Model):

  def __init__(self,
               feature_extractor=None,
               predictors_dict=None,
               is_training=True):
    super(MultiPredictorsRecognitionModel, self).__init__(
      feature_extractor,
      is_training)
    self._predictors_dict = predictors_dict
    self._is_training = is_training

    if len(self._predictors_dict) == 0:
      raise ValueError('predictors_list is empty!')

  def predict(self, preprocessed_inputs, scope=None):
    with tf.variable_scope(None, 'FeatureExtractor', [preprocessed_inputs]) as feat_scope:
      feature_maps = self._feature_extractor.extract_features(preprocessed_inputs, scope=feat_scope)
    predictions_dict = {}
    for name, predictor in self._predictors_dict.items():
      predictor_outputs = predictor.predict(feature_maps, scope='{}/Predictor'.format(name))
      predictions_dict.update({
        '{}/{}'.format(name, k) : v for k, v in predictor_outputs.items()
      })
    return predictions_dict

  def loss(self, predictions_dict, scope=None):
    with tf.variable_scope(scope, 'Loss', list(predictions_dict.values())):
      losses_dict = {}
      for name, predictor in self._predictors_dict.items():
        predictor_loss = predictor.loss({
          k.split('/')[1] : v
          for k, v in predictions_dict.items() if k.startswith('{}/'.format(name))
        }, scope='{}/Loss'.format(name))
        losses_dict[name] = predictor_loss
    return losses_dict

  def postprocess(self, predictions_dict, scope=None):
    with tf.variable_scope(scope, 'Postprocess', list(predictions_dict.values())):
      recognition_text_list = []
      recognition_scores_list = []
      for name, predictor in self._predictors_dict.items():
        predictor_outputs = predictor.postprocess({
          k.split('/')[1] : v
          for k, v in predictions_dict.items() if k.startswith('{}/'.format(name))
        }, scope='{}/Postprocess'.format(name))
        recognition_text_list.append(predictor_outputs['text'])
        recognition_scores_list.append(predictor_outputs['scores'])
      aggregated_recognition_dict = self._aggregate_recognition_results(
        recognition_text_list, recognition_scores_list)
    return aggregated_recognition_dict

  def provide_groundtruth(self, groundtruth_text_list, scope=None):
    with tf.variable_scope(scope, 'ProvideGroundtruth', [groundtruth_text_list]):
      batch_size = len(groundtruth_text_list)
      groundtruth_text = tf.stack(groundtruth_text_list, axis=0)
      for name, predictor in self._predictors_dict.items():
        predictor.provide_groundtruth(
          groundtruth_text,
          scope='{}/ProvideGroundtruth'.format(name))

  def _aggregate_recognition_results(self, text_list, scores_list, scope=None):
    """Aggregate recognition results by picking up ones with highest scores.
    Args
      text_list: a list of tensors with shape [batch_size]
      scores_list: a list of tensors with shape [batch_size]
    """
    with tf.variable_scope(scope, 'AggregateRecognitionResults', (text_list + scores_list)):
      stacked_text = tf.stack(text_list, axis=1)
      stacked_scores = tf.stack(scores_list, axis=1)
      argmax_scores = tf.argmax(stacked_scores, axis=1)
      batch_size = shape_utils.combined_static_and_dynamic_shape(stacked_text)[0]
      indices = tf.stack([tf.range(batch_size, dtype=tf.int64), argmax_scores], axis=1)
      aggregated_text = tf.gather_nd(stacked_text, indices)
      aggregated_scores = tf.gather_nd(stacked_scores, indices)
      recognition_dict = {'text': aggregated_text, 'scores': aggregated_scores}
    return recognition_dict
