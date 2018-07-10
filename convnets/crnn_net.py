import functools

import tensorflow as tf
from tensorflow.contrib.layers import conv2d, max_pool2d
from tensorflow.contrib.framework import arg_scope

from aster.core import convnet


class CrnnNet(convnet.Convnet):

  def _shape_check(self, preprocessed_inputs):
    preprocessed_inputs.get_shape().assert_has_rank(4)
    shape_assert = tf.Assert(
      tf.greater_equal(tf.shape(preprocessed_inputs)[1], 32),
      ['image height must be at least 32.']
    )
    return shape_assert

  def _extract_features(self, preprocessed_inputs):
    """Extract features
    Args:
      preprocessed_inputs: float32 tensor of shape [batch_size, image_height, image_width, 3]
    Return:
      feature_maps: a list of extracted feature maps
    """
    with arg_scope([conv2d], kernel_size=3, padding='SAME', stride=1), \
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
      feature_maps_dict = {
        'conv1': conv1, 'conv2': conv2, 'conv3': conv3, 'conv4': conv4,
        'conv5': conv5, 'conv6': conv6, 'conv7': conv7}
    return feature_maps_dict

  def _output_endpoints(self, feature_maps_dict):
    return [feature_maps_dict['conv7']]


class CrnnNetMultiBranches(CrnnNet):

  def __init__(self,
               conv_hyperparams=None,
               summarize_activations=False,
               is_training=True,
               use_branch_1=False,
               use_branch_2=False):
    super(CrnnNetMultiBranches, self).__init__(
      conv_hyperparams=conv_hyperparams,
      summarize_activations=summarize_activations,
      is_training=is_training
    )
    self._use_branch_1 = use_branch_1
    self._use_branch_2 = use_branch_2

  def _extract_features(self, preprocessed_inputs):
    with arg_scope([conv2d], kernel_size=3, padding='SAME', stride=1), \
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
      feature_maps_dict = {
        'conv1': conv1, 'conv2': conv2, 'conv3': conv3, 'conv4': conv4,
        'conv5': conv5, 'conv6': conv6, 'conv7': conv7}

      if self._use_branch_1:
        with tf.variable_scope('Branch1'):
          pool4_1 = max_pool2d(conv4, 2, stride=[2, 2], scope='pool4')
          conv5_1 = conv2d(pool4_1, 512, scope='conv5')
          conv6_1 = conv2d(conv5_1, 512, scope='conv6')
          pool6_1 = max_pool2d(conv6_1, 2, stride=[2, 1], scope='pool6')
          conv7_1 = conv2d(pool6_1, 512, kernel_size=[2, 1], padding='VALID', scope='conv7')
        feature_maps_dict.update({
          'branch1/pool4': pool4_1,
          'branch1/conv5': conv5_1,
          'branch1/conv6': conv6_1,
          'branch1/pool6': pool6_1,
          'branch1/conv7': conv7_1,
        })
      
      if self._use_branch_2:
        with tf.variable_scope('Branch2'):
          pool4_2 = max_pool2d(conv4, 2, stride=[2, 2], scope='pool4')
          conv5_2 = conv2d(pool4_2, 512, scope='conv5')
          conv6_2 = conv2d(conv5_2, 512, scope='conv6')
          pool6_2 = max_pool2d(conv6_2, 2, stride=[2, 2], scope='pool6')
          conv7_2 = conv2d(pool6_2, 512, kernel_size=[2, 1], padding='VALID', scope='conv7')
        feature_maps_dict.update({
          'branch2/pool4': pool4_2,
          'branch2/conv5': conv5_2,
          'branch2/conv6': conv6_2,
          'branch2/pool6': pool6_2,
          'branch2/conv7': conv7_2,
        })
    return feature_maps_dict


class CrnnNetTwoBranches(CrnnNetMultiBranches):
  def __init__(self,
               conv_hyperparams=None,
               summarize_activations=False,
               is_training=True):
    super(CrnnNetTwoBranches, self).__init__(
      conv_hyperparams=conv_hyperparams,
      summarize_activations=summarize_activations,
      is_training=is_training,
      use_branch_1=True,
      use_branch_2=False
    )
  
  def _output_endpoints(self, feature_maps_dict):
    return [
      feature_maps_dict['conv7'],
      feature_maps_dict['branch1/conv7'],
    ]


class CrnnNetThreeBranches(CrnnNetMultiBranches):
  def __init__(self,
               conv_hyperparams=None,
               summarize_activations=False,
               is_training=True):
    super(CrnnNetThreeBranches, self).__init__(
      conv_hyperparams=conv_hyperparams,
      summarize_activations=summarize_activations,
      is_training=is_training,
      use_branch_1=True,
      use_branch_2=True
    )
  
  def _output_endpoints(self, feature_maps_dict):
    return [
      feature_maps_dict['conv7'],
      feature_maps_dict['branch1/conv7'],
      feature_maps_dict['branch2/conv7'],
    ]


class CrnnNetTiny(convnet.Convnet):
  """For fast prototyping."""

  def _shape_check(self, preprocessed_inputs):
    preprocessed_inputs.get_shape().assert_has_rank(4)
    shape_assert = tf.Assert(
      tf.greater_equal(tf.shape(preprocessed_inputs)[1], 32),
      ['image height must be at least 32.']
    )
    return shape_assert

  def _extract_features(self, preprocessed_inputs):
    """Extract features
    Args:
      preprocessed_inputs: float32 tensor of shape [batch_size, image_height, image_width, 3]
    Return:
      feature_maps: a list of extracted feature maps
    """
    with arg_scope([conv2d], kernel_size=3, padding='SAME', stride=1), \
         arg_scope([max_pool2d], stride=2):
      conv1 = conv2d(preprocessed_inputs, 8, scope='conv1')
      pool1 = max_pool2d(conv1, 2, scope='pool1')
      conv2 = conv2d(pool1, 16, scope='conv2')
      pool2 = max_pool2d(conv2, 2, scope='pool2')
      conv3 = conv2d(pool2, 32, scope='conv3')
      conv4 = conv2d(conv3, 64, scope='conv4')
      pool4 = max_pool2d(conv4, 2, stride=[2, 1], scope='pool4')
      conv5 = conv2d(pool4, 128, scope='conv5')
      conv6 = conv2d(conv5, 128, scope='conv6')
      pool6 = max_pool2d(conv6, 2, stride=[2, 1], scope='pool6')
      conv7 = conv2d(pool6, 128, kernel_size=[2, 1], padding='VALID', scope='conv7')
      feature_maps_dict = {
        'conv1': conv1, 'conv2': conv2, 'conv3': conv3, 'conv4': conv4,
        'conv5': conv5, 'conv6': conv6, 'conv7': conv7}
    return feature_maps_dict

  def _output_endpoints(self, feature_maps_dict):
    return [feature_maps_dict['conv7']]
