import logging

import tensorflow as tf
from tensorflow.contrib.framework import arg_scope
from tensorflow.contrib.layers import fully_connected

from rare.core import model
from rare.utils import shape_utils


class CtcRecognitionModel(model.Model):

  def __init__(self,
               feature_extractor=None,
               label_map=None,
               fc_hyperparams=None,
               is_training=True):
    super(CtcRecognitionModel, self).__init__(feature_extractor, is_training)
    self._label_map = label_map
    self._fc_hyperparams = fc_hyperparams
    self._groundtruth_dict = {}

    logging.info('Number of classes: {}'.format(self.num_classes))

  @property
  def num_classes(self):
    # in tf.nn.ctc_loss, the largest label value is reserved for blank label
    return self._label_map.num_classes + 1

  def predict(self, preprocessed_inputs, scope=None):
    """
    Args:
      preprocessed_inputs: a float tensor with shape [batch_size, image_height, image_width, 3]
    Returns:
      predictions_dict: a diction of predicted tensors
    """
    with tf.variable_scope(scope, 'CtcRecognitionModel', [preprocessed_inputs]):
      with tf.variable_scope('FeatureExtractor') as feat_scope:
        feature_maps = self._feature_extractor.extract_features(
          preprocessed_inputs, scope=feat_scope)
        if len(feature_maps) != 1:
          raise ValueError('CtcRecognitionModel only accepts single feature sequence')
        feature_sequence = tf.squeeze(feature_maps[0], axis=1)

      with tf.variable_scope('Predictor'), \
           arg_scope(self._fc_hyperparams):
        logits = fully_connected(feature_sequence, self.num_classes, activation_fn=None)
    return {'logits': logits}

  def loss(self, predictions_dict, scope=None):
    with tf.variable_scope(scope, 'Loss', list(predictions_dict.values())):
      logits = predictions_dict['logits']
      batch_size, max_time, _ = shape_utils.combined_static_and_dynamic_shape(logits)
      losses = tf.nn.ctc_loss(
        tf.cast(self._groundtruth_dict['text_labels_sparse'], tf.int32),
        predictions_dict['logits'],
        tf.fill([batch_size], max_time),
        preprocess_collapse_repeated=False,
        ctc_merge_repeated=True,
        ignore_longer_outputs_than_inputs=True,
        time_major=False)
      loss = tf.reduce_mean(losses)
    return {'RecognitionLoss': loss}

  def postprocess(self, predictions_dict, scope=None):
    with tf.variable_scope(scope, 'Postprocess', list(predictions_dict.values())):
      logits = predictions_dict['logits']
      batch_size, max_time, _ = shape_utils.combined_static_and_dynamic_shape(logits)
      logits_time_major = tf.transpose(logits, [1,0,2])
      sparse_labels, log_prob = tf.nn.ctc_greedy_decoder(
        logits_time_major,
        tf.fill([batch_size], max_time),
        merge_repeated=True
      )
      labels = tf.sparse_tensor_to_dense(sparse_labels[0], default_value=-1)
      text = self._label_map.labels_to_text(labels)
      recognitions_dict = {'text': text}
    return recognitions_dict

  def provide_groundtruth(self, groundtruth_text_list, scope=None):
    with tf.variable_scope(scope, 'ProvideGroundtruth', groundtruth_text_list):
      groundtruth_text = tf.stack(groundtruth_text_list, axis=0)
      groundtruth_text_labels_sp, text_lengths = \
        self._label_map.text_to_labels(
          groundtruth_text, return_dense=False, return_lengths=True
        )
      self._groundtruth_dict['text_labels_sparse'] = groundtruth_text_labels_sp
      self._groundtruth_dict['text_lengths'] = text_lengths
