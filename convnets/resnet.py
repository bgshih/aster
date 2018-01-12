import functools

import tensorflow as tf
from tensorflow.contrib.layers import conv2d
from tensorflow.contrib.framework import arg_scope

from rare.core import convnet


class Resnet(convnet.Convnet):
  def __init__(self,
               conv_hyperparams=None,
               summarize_activations=None,
               is_training=None,
               resnet_spec=None):
    super(Resnet, self).__init__(
      conv_hyperparams=conv_hyperparams,
      summarize_activations=summarize_activations,
      is_training=is_training
    )
    self._resnet_spec = resnet_spec

  def _shape_check(self, preprocessed_inputs):
    preprocessed_inputs.get_shape().assert_has_rank(4)
    shape_assert = tf.Assert(
      tf.greater_equal(tf.shape(preprocessed_inputs)[1], 32),
      ['image height must be at least 32.']
    )
    return shape_assert

  def _residual_unit(self,
                     inputs,
                     num_outputs,
                     subsample=None,
                     is_training=True,
                     scope=None):
    with tf.variable_scope(scope, 'Unit', [inputs]):
      if subsample is None:
        conv1 = conv2d(inputs, num_outputs, kernel_size=1, stride=[1,1], scope='Conv1')
        shortcut = tf.identity(inputs, name='ShortCut')
      else:
        conv1 = conv2d(inputs, num_outputs, kernel_size=3, stride=subsample, scope='Conv1')
        shortcut = conv2d(inputs, num_outputs, kernel_size=3, stride=subsample, scope='ShortCut')
      conv2 = conv2d(conv1, num_outputs, kernel_size=3, stride=[1,1], activation_fn=None, scope='Conv2')
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
      conv_0 = conv2d(inputs, 32, scope='Conv0')
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
    outputs_dict = {}
    for i, block_output in enumerate(block_outputs_list):
      outputs_dict['Block_{}'.format(i)] = block_output
    return outputs_dict

  def _extract_features(self, preprocessed_inputs):
    return self._resnet(preprocessed_inputs, is_training=self._is_training)

  def _output_endpoints(self, feature_maps_dict):
    return [feature_maps_dict['Block_5']]


class Resnet50Layer(Resnet):

  def __init__(self,
               conv_hyperparams=None,
               summarize_activations=None,
               is_training=None):
    # block_name: (scope, num_units, num_outputs, first_subsample)
    resnet_spec = [
      ('Block_1', 3, 32, [2, 2]),
      ('Block_2', 4, 64, [2, 2]),
      ('Block_3', 6, 128, [2, 1]),
      ('Block_4', 6, 256, [2, 1]),
      ('Block_5', 3, 512, [2, 1]),
    ]
    super(Resnet50Layer, self).__init__(
      conv_hyperparams=conv_hyperparams,
      summarize_activations=summarize_activations,
      is_training=is_training,
      resnet_spec=resnet_spec
    )


class ResnetForSTN(Resnet):

  def __init__(self,
               conv_hyperparams=None,
               summarize_activations=None,
               is_training=None):
    resnet_spec = [
      ('Block_1', 1, 8, [2, 2]), # => [32,64]
      ('Block_2', 1, 16, [2, 2]), # => [16,32]
      ('Block_3', 1, 32, [2, 2]), # => [8,16]
      ('Block_4', 1, 64, [2, 2]), # => [4,8]
      ('Block_5', 1, 64, [2, 2]), # => [2,4]
    ]
    super(ResnetForSTN, self).__init__(
      conv_hyperparams=conv_hyperparams,
      summarize_activations=summarize_activations,
      is_training=is_training,
      resnet_spec=resnet_spec
    )
