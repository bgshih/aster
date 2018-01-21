import logging
import functools

import tensorflow as tf
from tensorflow.contrib import seq2seq

from rare.core import model
from rare.core import standard_fields as fields
from rare.c_ops import ops
from rare.utils import shape_utils


class MultiPredictorsRecognitionModel(model.Model):

  def __init__(self,
               spatial_transformer=None,
               feature_extractor=None,
               predictors_dict=None,
               regression_loss=None,
               keypoint_supervision=None,
               is_training=True):
    super(MultiPredictorsRecognitionModel, self).__init__(
      feature_extractor,
      is_training)
    self._spatial_transformer = spatial_transformer
    self._predictors_dict = predictors_dict
    self._regression_loss = regression_loss
    self._keypoint_supervision = keypoint_supervision
    self._is_training = is_training

    if len(self._predictors_dict) == 0:
      raise ValueError('predictors_list is empty!')
    self._groundtruth_dict = {}

  def predict(self, resized_images, scope=None):
    predictions_dict = {}
    if self._spatial_transformer:
      stn_inputs = self.preprocess(resized_images)
      transform_output_dict = self._spatial_transformer.batch_transform(stn_inputs)
      preprocessed_inputs = transform_output_dict['rectified_images']
      control_points = transform_output_dict['control_points']
      predictions_dict.update({
        'control_points': control_points,
        'rectified_images': transform_output_dict['rectified_images']
      })
    else:
      preprocessed_inputs = self.preprocess(resized_images)
    with tf.variable_scope(None, 'FeatureExtractor', [preprocessed_inputs]) as feat_scope:
      feature_maps = self._feature_extractor.extract_features(preprocessed_inputs, scope=feat_scope)
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
      
      if self._spatial_transformer is not None and self._keypoint_supervision == True:
        control_points = predictions_dict['control_points']
        num_control_points = tf.shape(control_points)[1]
        masked_control_points = tf.boolean_mask(
          control_points,
          self._groundtruth_dict['control_points_mask'])
        flat_masked_control_points = tf.reshape(
          masked_control_points,
          [-1, 2 * self._spatial_transformer._num_control_points]
        )
        losses_dict['KeypointSupervisionLoss'] = self._regression_loss(
          flat_masked_control_points,
          self._groundtruth_dict['control_points'])
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

  def provide_groundtruth(self, groundtruth_lists, scope=None):
    with tf.variable_scope(scope, 'ProvideGroundtruth', list(groundtruth_lists.values())):
      # provide groundtruth_text to all predictors
      groundtruth_text = tf.stack(
        groundtruth_lists[fields.InputDataFields.groundtruth_text], axis=0)
      for name, predictor in self._predictors_dict.items():
        predictor.provide_groundtruth(
          groundtruth_text,
          scope='{}/ProvideGroundtruth'.format(name))

      # provide groundtruth keypoints
      if self._spatial_transformer is not None and self._keypoint_supervision == True:
        groundtruth_keypoints_lengths = tf.stack([
          tf.shape(keypoints)[0] for keypoints in
          groundtruth_lists[fields.InputDataFields.groundtruth_keypoints]
        ])
        max_keypoints_length = tf.reduce_max(groundtruth_keypoints_lengths)
        has_groundtruth_keypoints = tf.equal(groundtruth_keypoints_lengths, max_keypoints_length)
        groundtruth_keypoints = tf.stack([
          tf.pad(keypoints, [[0, max_keypoints_length - tf.shape(keypoints)[0]]])
          for keypoints in groundtruth_lists[fields.InputDataFields.groundtruth_keypoints]
        ])
        groundtruth_keypoints = tf.boolean_mask(groundtruth_keypoints, has_groundtruth_keypoints)
        groundtruth_control_points = ops.divide_curve(
          groundtruth_keypoints,
          num_key_points=self._spatial_transformer._num_control_points)
        self._groundtruth_dict['control_points_mask'] = has_groundtruth_keypoints
        self._groundtruth_dict['control_points'] = groundtruth_control_points

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
