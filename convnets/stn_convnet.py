import functools

import tensorflow as tf
from tensorflow.contrib.layers import conv2d, max_pool2d
from tensorflow.contrib.framework import arg_scope

from rare.core import convnet


class StnConvnet(convnet.Convnet):

  def _extract_features(self, preprocessed_inputs):
    """Extract features
    Args:
      preprocessed_inputs: float32 tensor of shape [batch_size, image_height, image_width, 3]
    Return:
      feature_maps: a list of extracted feature maps
    """
    with arg_scope([conv2d], kernel_size=3, activation_fn=tf.nn.relu), \
         arg_scope([max_pool2d], kernel_size=2, stride=2):
      conv1 = conv2d(preprocessed_inputs, 32, scope='conv1') # 64
      pool1 = max_pool2d(conv1, scope='pool1')
      conv2 = conv2d(pool1, 64, scope='conv2')  # 32
      pool2 = max_pool2d(conv2, scope='pool2')
      conv3 = conv2d(pool2, 128, scope='conv3')  # 16
      pool3 = max_pool2d(conv3, scope='pool3')
      conv4 = conv2d(pool3, 256, scope='conv4')  # 8
      pool4 = max_pool2d(conv4, scope='pool4')
      conv5 = conv2d(pool4, 256, scope='conv5')  # 4
      pool5 = max_pool2d(conv5, scope='pool5')
      conv6 = conv2d(pool5, 256, scope='conv6')  # 2
      feature_maps_dict = { 'conv6': conv6 }
    return feature_maps_dict

  def _output_endpoints(self, feature_maps_dict):
    return [feature_maps_dict['conv6']]
