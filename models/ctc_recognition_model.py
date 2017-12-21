import logging

import tensorflow as tf
from tensorflow.contrib.framework import arg_scope
from tensorflow.contrib.layers import fully_connected

from rare.core import hyperparams
from rare.core import label_map
from rare.core import feature_extractor
from rare.core import bidirectional_rnn
from rare.models import model, model_pb2
from rare.utils import shape_utils


class CtcRecognitionModel(model.Model):

  def __init__(self,
               feature_extractor=None,
               bidirectional_rnn_list=None,
               fc_hyperparams=None,
               label_map=None,
               is_training=True):
    super(CtcRecognitionModel, self).__init__(feature_extractor, is_training)
    self._bidirectional_rnn_list = bidirectional_rnn_list
    self._label_map = label_map
    self._fc_hyperparams = fc_hyperparams
    self._groundtruth_dict = {}

    logging.info('Number of classes: {}'.format(self.num_classes))

  @property
  def num_classes(self):
    # in tf.nn.ctc_loss, the largest label value is reserved for blank label
    return self._label_map.num_classes + 1

  def preprocess(self, resized_inputs, scope=None):
    if resized_inputs.dtype is not tf.float32:
      raise ValueError('`preprocess` expects a tf.float32 tensor')
    with tf.variable_scope(scope, 'ModelPreprocess', [resized_inputs]) as preprocess_scope:
      preprocess_inputs = self._feature_extractor.preprocess(resized_inputs, scope=preprocess_scope)
    return preprocess_inputs

  def predict(self, preprocessed_inputs, scope=None):
    """
    Args:
      preprocessed_inputs: a float tensor with shape [batch_size, image_height, image_width, 3]
    Returns:
      predictions_dict: a diction of predicted tensors
    """
    with tf.variable_scope(scope, 'CtcRecognitionModel', [preprocessed_inputs]):
      with tf.variable_scope('FeatureExtractor') as feat_scope:
        feature_map = self._feature_extractor.extract_features(
          preprocessed_inputs, scope=feat_scope)[0]

      with tf.variable_scope('Predictor'):
        feature_map_shape = shape_utils.combined_static_and_dynamic_shape(feature_map)
        batch_size, map_depth = feature_map_shape[0], feature_map_shape[3]
        if batch_size is None or map_depth is None:
          raise ValueError('batch_size and map_depth must be static')
        feature_sequence = tf.reshape(feature_map, [batch_size, -1, map_depth])

        last_outputs = feature_sequence
        for i, brnn in enumerate(self._bidirectional_rnn_list):
          last_outputs = brnn.predict(last_outputs, scope='BidirectionalRnn_{}'.format(i+1))

        with arg_scope(self._fc_hyperparams):
          logits = fully_connected(last_outputs, self.num_classes, activation_fn=None)
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


def build(config, is_training):
  if not isinstance(config, model_pb2.CtcRecognitionModel):
    raise ValueError('config not of type model_pb2.CtcRecognitionModel')

  feature_extractor_object = feature_extractor.build(
    config.feature_extractor,
    is_training=is_training
  )
  label_map_object = label_map.build(config.label_map)
  bidirectional_rnn_list = [
    bidirectional_rnn.build(brnn_config, is_training) for brnn_config in config.bidirectional_rnn
  ]
  fc_hyperparams_object = hyperparams.build(config.fc_hyperparams, is_training)
  model_object = CtcRecognitionModel(
    feature_extractor=feature_extractor_object,
    bidirectional_rnn_list=bidirectional_rnn_list,
    fc_hyperparams=fc_hyperparams_object,
    label_map=label_map_object,
    is_training=is_training)
  return model_object
