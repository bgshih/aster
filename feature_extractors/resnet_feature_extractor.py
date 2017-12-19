import tensorflow as tf
from tensorflow.contrib.layers import conv2d
from tensorflow.contrib.framework import arg_scope


class ResnetFeatureExtractor(object):

  def __init__(self,
               conv_hyperparams=None,
               summarize_inputs=False,
               is_training=True,
               resnet_spec=None):
    self._conv_hyperparams = conv_hyperparams
    self._summarize_inputs = summarize_inputs
    self._is_training = is_training
    self._resnet_spec = resnet_spec

  def preprocess(self, resized_inputs, scope=None):
    with tf.variable_scope(scope, 'FeatureExtractorPreprocess', [resized_inputs]):
      preprocessed_inputs = (2.0 / 255.0) * resized_inputs - 1.0
    return preprocessed_inputs

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
      unit_output = self._residual_unit(inputs, num_outputs, subsample=first_subsample, is_training=is_training, scope='Block_1')
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


class Resnet52LayerFeatureExtractor(ResnetFeatureExtractor):

  def __init__(self,
               conv_hyperparams=None,
               summarize_inputs=False,
               is_training=True):
    # block_name: (scope, num_units, num_outputs, first_subsample)
    resnet_spec = [
      ('Block_1', 6, 16, [2, 2]),
      ('Block_2', 6, 32, [2, 2]),
      ('Block_3', 6, 64, [2, 1]),
      ('Block_4', 6, 128, [2, 1]),
      ('Block_5', 6, 256, [2, 1]),
    ]
    super(Resnet52LayerFeatureExtractor, self).__init__(
      conv_hyperparams, summarize_inputs, is_training,
      resnet_spec=resnet_spec)
