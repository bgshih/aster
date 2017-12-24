import functools

import tensorflow as tf
from tensorflow.contrib.layers import conv2d, max_pool2d
from tensorflow.contrib.framework import arg_scope

from rare.core import hyperparams
from rare.core import bidirectional_rnn
from rare.core import feature_extractor, feature_extractor_pb2


class BaselineFeatureExtractor(feature_extractor.FeatureExtractor):

  def __init__(self,
               conv_hyperparams=None,
               summarize_inputs=False,
               brnn_fn_list=[],
               is_training=None):
    super(BaselineFeatureExtractor, self).__init__(
      summarize_inputs=summarize_inputs,
      brnn_fn_list=brnn_fn_list,
      is_training=is_training)
    self._conv_hyperparams = conv_hyperparams # FIXME: add it back
    self._summarize_inputs = summarize_inputs

  def _extract_features(self, preprocessed_inputs):
    """Extract features
    Args:
      preprocessed_inputs: float32 tensor of shape [batch_size, image_height, image_width, 3]
    Return:
      feature_maps: a list of extracted feature maps
    """
    preprocessed_inputs.get_shape().assert_has_rank(4)
    shape_assert = tf.Assert(
      tf.greater_equal(tf.shape(preprocessed_inputs)[1], 32),
      ['image height must be at least 32.']
    )
    if self._summarize_inputs:
      tf.summary.histogram('preprocessed_inputs', preprocessed_inputs)
    with tf.control_dependencies([shape_assert]), \
         arg_scope(self._conv_hyperparams), \
         arg_scope([conv2d], kernel_size=3, padding='SAME', stride=1), \
         arg_scope([max_pool2d], stride=2):
      conv1 = conv2d(preprocessed_inputs, 64, scope='conv1')
      pool1 = max_pool2d(conv1, 2, scope='pool1')
      conv2 = conv2d(pool1, 128, scope='conv2')
      pool2 = max_pool2d(conv2, 2, scope='pool2')
      conv3 = conv2d(pool2, 256, scope='conv3')
      conv4 = conv2d(conv3, 256, scope='conv4')
      pool4 = max_pool2d(conv4, 2, stride=[2, 1], scope='pool4')
      conv5 = conv2d(pool4, 512, scope='conv5')
      conv6 = conv2d(conv5, 512, scope='conv6')
      pool6 = max_pool2d(conv6, 2, stride=[2, 1], scope='pool6')
      conv7 = conv2d(pool6, 512, kernel_size=[2, 1], padding='VALID', scope='conv7')
    if self._summarize_inputs:
      for layer in [conv1, pool1, conv2, pool2, conv3,
                    conv4, pool4, conv5, conv6, pool6, conv7]:
        tf.summary.histogram(layer.op.name, layer)
    return [conv7]


class BaselineTwoBranchFeatureExtractor(BaselineFeatureExtractor):

  def _extract_features(self, preprocessed_inputs):
    preprocessed_inputs.get_shape().assert_has_rank(4)
    shape_assert = tf.Assert(
      tf.greater_equal(tf.shape(preprocessed_inputs)[1], 32),
      ['image height must be at least 32.']
    )
    if self._summarize_inputs:
      tf.summary.histogram('preprocessed_inputs', preprocessed_inputs)
    with tf.control_dependencies([shape_assert]), \
         arg_scope(self._conv_hyperparams), \
         arg_scope([conv2d], kernel_size=3, padding='SAME', stride=1), \
         arg_scope([max_pool2d], stride=2):
      conv1 = conv2d(preprocessed_inputs, 64, scope='conv1')
      pool1 = max_pool2d(conv1, 2, scope='pool1')
      conv2 = conv2d(pool1, 128, scope='conv2')
      pool2 = max_pool2d(conv2, 2, scope='pool2')
      conv3 = conv2d(pool2, 256, scope='conv3')
      conv4 = conv2d(conv3, 256, scope='conv4')
      with tf.variable_scope('Branch1'):
        pool4 = max_pool2d(conv4, 2, stride=[2, 1], scope='pool4')
        conv5 = conv2d(pool4, 512, scope='conv5')
        conv6 = conv2d(conv5, 512, scope='conv6')
        pool6 = max_pool2d(conv6, 2, stride=[2, 1], scope='pool6')
        conv7_1 = conv2d(pool6, 512, kernel_size=[2, 1], padding='VALID', scope='conv7')
      with tf.variable_scope('Branch2'):
        pool4 = max_pool2d(conv4, 2, stride=[2, 2], scope='pool4')
        conv5 = conv2d(pool4, 512, scope='conv5')
        conv6 = conv2d(conv5, 512, scope='conv6')
        pool6 = max_pool2d(conv6, 2, stride=[2, 1], scope='pool6')
        conv7_2 = conv2d(pool6, 512, kernel_size=[2, 1], padding='VALID', scope='conv7')
    return [conv7_1, conv7_2]


class BaselineThreeBranchFeatureExtractor(BaselineFeatureExtractor):

  def _extract_features(self, preprocessed_inputs):
    preprocessed_inputs.get_shape().assert_has_rank(4)
    shape_assert = tf.Assert(
      tf.greater_equal(tf.shape(preprocessed_inputs)[1], 32),
      ['image height must be at least 32.']
    )
    if self._summarize_inputs:
      tf.summary.histogram('preprocessed_inputs', preprocessed_inputs)
    with tf.control_dependencies([shape_assert]), \
         arg_scope(self._conv_hyperparams), \
         arg_scope([conv2d], kernel_size=3, padding='SAME', stride=1), \
         arg_scope([max_pool2d], stride=2):
      conv1 = conv2d(preprocessed_inputs, 64, scope='conv1')
      pool1 = max_pool2d(conv1, 2, scope='pool1')
      conv2 = conv2d(pool1, 128, scope='conv2')
      pool2 = max_pool2d(conv2, 2, scope='pool2')
      conv3 = conv2d(pool2, 256, scope='conv3')
      conv4 = conv2d(conv3, 256, scope='conv4')
      with tf.variable_scope('Branch1'):
        pool4 = max_pool2d(conv4, 2, stride=[2, 1], scope='pool4')
        conv5 = conv2d(pool4, 512, scope='conv5')
        conv6 = conv2d(conv5, 512, scope='conv6')
        pool6 = max_pool2d(conv6, 2, stride=[2, 1], scope='pool6')
        conv7_1 = conv2d(pool6, 512, kernel_size=[2, 1], padding='VALID', scope='conv7')
      with tf.variable_scope('Branch2'):
        pool4 = max_pool2d(conv4, 2, stride=[2, 2], scope='pool4')
        conv5 = conv2d(pool4, 512, scope='conv5')
        conv6 = conv2d(conv5, 512, scope='conv6')
        pool6 = max_pool2d(conv6, 2, stride=[2, 1], scope='pool6')
        conv7_2 = conv2d(pool6, 512, kernel_size=[2, 1], padding='VALID', scope='conv7')
      with tf.variable_scope('Branch3'):
        pool4 = max_pool2d(conv4, 2, stride=[2, 2], scope='pool4')
        conv5 = conv2d(pool4, 512, scope='conv5')
        conv6 = conv2d(conv5, 512, scope='conv6')
        pool6 = max_pool2d(conv6, 2, stride=[2, 2], scope='pool6')
        conv7_3 = conv2d(pool6, 512, kernel_size=[2, 1], padding='VALID', scope='conv7')
    return [conv7_1, conv7_2, conv7_3]


def build(config, is_training):
  if not isinstance(config, feature_extractor_pb2.BaselineFeatureExtractor):
    raise ValueError('config is not of type feature_extractor_pb2.BaselineFeatureExtractor')

  if config.baseline_type == feature_extractor_pb2.BaselineFeatureExtractor.SINGLE_BRANCH:
    baseline_feature_extractor_class = BaselineFeatureExtractor
  elif config.baseline_type == feature_extractor_pb2.BaselineFeatureExtractor.TWO_BRANCH:
    baseline_feature_extractor_class = BaselineTwoBranchFeatureExtractor
  elif config.baseline_type == feature_extractor_pb2.BaselineFeatureExtractor.THREE_BRANCH:
    baseline_feature_extractor_class = BaselineThreeBranchFeatureExtractor
  else:
    raise ValueError('Unknown baseline feature extractor type: {}'.format(config.baseline_type))

  brnn_fn_list_object = [
    functools.partial(bidirectional_rnn.build, brnn_config, is_training)
    for brnn_config in config.bidirectional_rnn
  ]
  return baseline_feature_extractor_class(
    conv_hyperparams=hyperparams.build(config.conv_hyperparams, is_training),
    summarize_inputs=config.summarize_inputs,
    brnn_fn_list=brnn_fn_list_object
  )
