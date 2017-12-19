import tensorflow as tf
from tensorflow.contrib.layers import conv2d, max_pool2d
from tensorflow.contrib.framework import arg_scope


class BaselineFeatureExtractor(object):

  def __init__(self,
               conv_hyperparams=None,
               summarize_inputs=False):
    self._conv_hyperparams = conv_hyperparams # FIXME: add it back
    self._summarize_inputs = summarize_inputs

  def preprocess(self, resized_inputs):
    preprocessed_inputs = (2.0 / 255.0) * resized_inputs - 1.0
    return preprocessed_inputs

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
