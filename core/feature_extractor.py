from abc import ABCMeta
from abc import abstractmethod

import tensorflow as tf
from tensorflow.contrib.layers import conv2d, max_pool2d
from tensorflow.contrib.framework import arg_scope

from rare.core import feature_extractor_pb2
from rare.core import hyperparams


class FeatureExtractor(object):
  """Abstract class for feature extractor."""
  __metaclass__ = ABCMeta

  def __init__(self,
               summarize_inputs=False,
               is_training=True):
    self._summarize_inputs = summarize_inputs
    self._is_training = is_training

  def preprocess(self, resized_inputs, scope=None):
    with tf.variable_scope(scope, 'FeatureExtractorPreprocess', [resized_inputs]):
      preprocessed_inputs = (2.0 / 255.0) * resized_inputs - 1.0
    return preprocessed_inputs

  @abstractmethod
  def extract_features(self, preprocessed_inputs, scope=None):
    pass


def build(config, is_training):
  if not isinstance(config, feature_extractor_pb2.FeatureExtractor):
    raise ValueError('config not of type '
                     'feature_extractor_pb2.FeatureExtractor')

  feature_extractor_oneof = config.WhichOneof('feature_extractor_oneof')
  if feature_extractor_oneof == 'baseline_feature_extractor':
    return _build_baseline(config.baseline_feature_extractor, is_training)
  elif feature_extractor_oneof == 'resnet_feature_extractor':
    return _build_resnet(config.resnet_feature_extractor, is_training)
  else:
    raise ValueError('Unknown feature_extractor_oneof: {}'.format(feature_extractor_oneof))


class ResnetFeatureExtractor(FeatureExtractor):
  def __init__(self,
               summarize_inputs=None,
               is_training=None,
               conv_hyperparams=None,
               resnet_spec=None):
    super(ResnetFeatureExtractor, self).__init__(
      summarize_inputs=summarize_inputs,
      is_training=is_training
    )
    self._conv_hyperparams = conv_hyperparams
    self._resnet_spec = resnet_spec

  def _residual_unit(self,
                     inputs,
                     num_outputs,
                     subsample=None,
                     is_training=True,
                     scope=None):
    with tf.variable_scope(scope, 'Unit', [inputs]), \
         arg_scope([conv2d], kernel_size=3):
      if subsample is None:
        conv1 = conv2d(inputs, num_outputs, stride=[1,1], scope='Conv1')
        shortcut = tf.identity(inputs, name='ShortCut')
      else:
        conv1 = conv2d(inputs, num_outputs, stride=subsample, scope='Conv1')
        shortcut = conv2d(inputs, num_outputs, stride=subsample, scope='ShortCut')
      conv2 = conv2d(conv1, num_outputs, stride=[1,1], activation_fn=None, scope='Conv2')
      output = tf.nn.relu(tf.add(conv2, shortcut))
    return output

  def _residual_block(self,
                      inputs,
                      num_outputs,
                      num_units,
                      first_subsample=None,
                      is_training=True,
                      scope=None):
    with tf.variable_scope(scope, 'Block', [inputs]):
      unit_output = self._residual_unit(inputs, num_outputs, subsample=first_subsample, is_training=is_training, scope='Unit_1')
      for i in range(1, num_units):
        unit_output = self._residual_unit(unit_output, num_outputs, subsample=None, scope='Unit_{}'.format(i+1))
    return unit_output

  def _resnet(self, inputs, is_training=True, scope=None):
    with tf.variable_scope(scope, 'ResNet', [inputs]), \
         arg_scope(self._conv_hyperparams), \
         arg_scope([conv2d], kernel_size=3, padding='SAME', stride=1):
      conv_0 = conv2d(inputs, 16, scope='Conv0')
      block_outputs_list = [conv_0]
      for (scope, num_units, num_outputs, first_subsample) in self._resnet_spec:
        block_output = self._residual_block(
          block_outputs_list[-1],
          num_outputs,
          num_units,
          first_subsample=first_subsample,
          is_training=is_training,
          scope=scope
        )
        block_outputs_list.append(block_output)
    return block_outputs_list

  def extract_features(self, preprocessed_inputs, scope=None):
    with tf.variable_scope(scope, 'FeatureExtractor', [preprocessed_inputs]):
      preprocessed_inputs.get_shape().assert_has_rank(4)
      shape_assert = tf.Assert(
        tf.greater_equal(tf.shape(preprocessed_inputs)[1], 32),
        ['image height must be at least 32.']
      )
      if self._summarize_inputs:
        tf.summary.histogram('preprocessed_inputs', preprocessed_inputs)
      with tf.control_dependencies([shape_assert]):
        resnet_outputs = self._resnet(preprocessed_inputs, is_training=self._is_training)
      if self._summarize_inputs:
        for output in resnet_outputs:
          tf.summary.histogram(output.op.name, output)
    return [resnet_outputs[-1]]


class Resnet52LayerFeatureExtractor(ResnetFeatureExtractor):

  def __init__(self,
               summarize_inputs=None,
               is_training=None,
               conv_hyperparams=None):
    # block_name: (scope, num_units, num_outputs, first_subsample)
    resnet_spec = [
      ('Block_1', 6, 16, [2, 2]),
      ('Block_2', 6, 32, [2, 2]),
      ('Block_3', 6, 64, [2, 1]),
      ('Block_4', 6, 128, [2, 1]),
      ('Block_5', 6, 256, [2, 1]),
    ]
    super(Resnet52LayerFeatureExtractor, self).__init__(
      summarize_inputs,
      is_training,
      conv_hyperparams=conv_hyperparams,
      resnet_spec=resnet_spec
    )


def _build_resnet(config, is_training):
  if not isinstance(config, feature_extractor_pb2.ResnetFeatureExtractor):
    raise ValueError('config is not of type feature_extractor_pb2.ResnetFeatureExtractor')

  resnet_type = config.resnet_type
  if resnet_type == feature_extractor_pb2.ResnetFeatureExtractor.RESNET_50:
    resnet_class = Resnet52LayerFeatureExtractor
  else:
    raise ValueError('Unknown resnet type: {}'.format(resnet_type))

  conv_hyperparams = hyperparams.build(
    config.conv_hyperparams,
    is_training)
  return resnet_class(
    summarize_inputs=config.summarize_inputs,
    is_training=is_training,
    conv_hyperparams=conv_hyperparams
  )


class BaselineFeatureExtractor(FeatureExtractor):

  def __init__(self,
               conv_hyperparams=None,
               summarize_inputs=False):
    super(BaselineFeatureExtractor, self).__init__()
    self._conv_hyperparams = conv_hyperparams # FIXME: add it back
    self._summarize_inputs = summarize_inputs

  def preprocess(self, resized_inputs, scope=None):
    with tf.variable_scope(scope, 'ModelPreprocess', [resized_inputs]):
      preprocessed_inputs = (2.0 / 255.0) * resized_inputs - 1.0
    return preprocessed_inputs

  def extract_features(self, preprocessed_inputs, scope=None):
    """Extract features
    Args:
      preprocessed_inputs: float32 tensor of shape [batch_size, image_height, image_width, 3]
    Return:
      feature_maps: a list of extracted feature maps
    """
    with tf.variable_scope(scope, 'FeatureExtractor', [preprocessed_inputs]):
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
           arg_scope([max_pool2d], stride=2), \
           tf.variable_scope(scope, 'FeatureExtractor'):
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
            tf.summary.histogram(layer.name, layer)
    return [conv7]


def _build_baseline(config, is_training):
  if not isinstance(config, feature_extractor_pb2.BaselineFeatureExtractor):
    raise ValueError('config is not of type feature_extractor_pb2.BaselineFeatureExtractor')
  return BaselineFeatureExtractor(
    conv_hyperparams=hyperparams.build(config.conv_hyperparams, is_training),
    summarize_inputs=config.summarize_inputs
  )
