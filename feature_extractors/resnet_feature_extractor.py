import functools

import tensorflow as tf
from tensorflow.contrib.layers import conv2d
from tensorflow.contrib.framework import arg_scope

from rare.core import hyperparams
from rare.core import bidirectional_rnn
from rare.core import feature_extractor, feature_extractor_pb2


class ResnetFeatureExtractor(feature_extractor.FeatureExtractor):
  def __init__(self,
               summarize_inputs=None,
               brnn_fn_list=[],
               is_training=None,
               conv_hyperparams=None,
               resnet_spec=None):
    super(ResnetFeatureExtractor, self).__init__(
      summarize_inputs=summarize_inputs,
      brnn_fn_list=brnn_fn_list,
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

  def _extract_features(self, preprocessed_inputs):
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


class Resnet50LayerFeatureExtractor(ResnetFeatureExtractor):

  def __init__(self,
               summarize_inputs=None,
               brnn_fn_list=[],
               is_training=None,
               conv_hyperparams=None):
    # block_name: (scope, num_units, num_outputs, first_subsample)
    resnet_spec = [
      ('Block_1', 3, 32, [2, 2]),
      ('Block_2', 4, 64, [2, 2]),
      ('Block_3', 6, 128, [2, 1]),
      ('Block_4', 6, 256, [2, 1]),
      ('Block_5', 3, 512, [2, 1]),
    ]
    super(Resnet50LayerFeatureExtractor, self).__init__(
      summarize_inputs=summarize_inputs,
      brnn_fn_list=[],
      is_training=is_training,
      conv_hyperparams=conv_hyperparams,
      resnet_spec=resnet_spec
    )


def build(config, is_training):
  if not isinstance(config, feature_extractor_pb2.ResnetFeatureExtractor):
    raise ValueError('config is not of type feature_extractor_pb2.ResnetFeatureExtractor')

  resnet_type = config.resnet_type
  if resnet_type == feature_extractor_pb2.ResnetFeatureExtractor.RESNET_50:
    resnet_class = Resnet50LayerFeatureExtractor
  else:
    raise ValueError('Unknown resnet type: {}'.format(resnet_type))

  conv_hyperparams = hyperparams.build(
    config.conv_hyperparams,
    is_training)
  brnn_fn_list_object = [
    functools.partial(bidirectional_rnn.build, brnn_config, is_training)
    for brnn_config in config.bidirectional_rnn
  ]
  return resnet_class(
    summarize_inputs=config.summarize_inputs,
    brnn_fn_list=brnn_fn_list_object,
    is_training=is_training,
    conv_hyperparams=conv_hyperparams,
  )
