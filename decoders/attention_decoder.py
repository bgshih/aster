import tensorflow as tf
from tf.contrib.layers import conv2d, fully_connected


class AttentionDecoder():
  def __init__(self,
               num_attention_units,
               attention_conv_kernel_size):
    pass

  def predict(self, feature_map, num_steps):
    if isinstance(feature_map, list):
      feature_map = feature_map[-1]

      initial_attention = tf.zeros([])

  def _predict_step(self, feature_map, last_state, last_attention, last_output, reuse=None):
    """
    Args:
      feature_map: a float32 tensor with shape [batch_size, map_height, map_width, depth]
      last_state: a float32 tensor with shape [batch_size, ]
    """

    batch_size = 

    vh = conv2d(
      feature_map,
      self._num_attention_units,
      kernel_size=1,
      stride=1,
      biases_initializer=None
    ) # => [batch_size, map_height, map_width, num_attention_units]
    last_attention_conv = conv2d(
      last_attention,
      self._num_attention_units,
      kernel_size=self._attention_conv_kernel_size,
      stride=1,
      biases_initializer=None,
      reuse=reuse
    ) # => [1, map_height, map_width, num_attention_units]
    ws = fully_connected(
      last_state,
      self._num_attention_units,
      activation_fn=None,
      biases_initializer=tf.zeros_initializer(),
    ) # => [1, num_attention_units], bias is added
    attention_sum = tf.bias_add(
      tf.add(vh, last_attention_conv),
      ws
    ) # => [1, map_height, map_width, num_attention_units]
    attention_scores = slim.conv2d(
      tf.tanh(att_sum),
      1,
      activation_fn=None,
      biases_initializer=None,
      reuse=reuse
    ) # => [1, map_height, map_width, 1]
    attention_weights = tf.nn.softmax(
      tf.reshape(tf.squeeze(attention_scores, axis=3),
                 [batch_size, -1, 1]), # => [1, map_height * map_width, 1]
      dim=1
    ) # => [1, map_height * map_width, 1]
    feature_flat = tf.reshape(
      feature_map,
      [num_heads, -1, feature_map_depth]
    ) # [1, map_height * map_width, map_depth]
    glimpse = tf.squeeze(
      tf.matmul(feature_flat_trans, feature_flat),
      axis=2
    ) # [1, map_depth]

    return self._rnn(glimpse, last_state)
    