import tensorflow as tf
from tensorflow.contrib.layers import conv2d, max_pool2d
from tensorflow.contrib.framework import arg_scope


class BaselineFeatureExtractor(object):

  def __init__(self,
               conv_hyperparams=None):
    self._conv_hyperparams = conv_hyperparams # FIXME: add it back

  def preprocess(self, resized_inputs):
    return (2.0 / 255.0) * resized_inputs - 1.0

  def extract_features(self, preprocessed_inputs, scope=None):
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

    with tf.control_dependencies([shape_assert]), \
         arg_scope([conv2d], kernel_size=3, padding='SAME', stride=1), \
         arg_scope([max_pool2d], stride=2), \
         tf.variable_scope(scope, 'FeatureExtractor'):

      conv1 = conv2d(preprocessed_inputs, 64)
      pool1 = max_pool2d(conv1, 2)
      conv2 = conv2d(pool1, 128)
      pool2 = max_pool2d(conv2, 2)
      conv3 = conv2d(pool2, 256)
      conv4 = conv2d(conv3, 256)
      pool4 = max_pool2d(conv4, 2, stride=[2, 1])
      conv5 = conv2d(pool4, 512)
      conv6 = conv2d(conv5, 512)
      pool6 = max_pool2d(conv6, 2, stride=[2, 1])
      conv7 = conv2d(pool6, 512, kernel_size=[2, 1], padding='VALID')

      # print('conv1', conv1)
      # print('pool1', pool1)
      # print('conv2', conv2)
      # print('pool2', pool2)
      # print('conv3', conv3)
      # print('conv4', conv4)
      # print('pool4', pool4)
      # print('conv5', conv5)
      # print('conv6', conv6)
      # print('pool6', pool6)
      # print('conv7', conv7)

    return [conv7]
