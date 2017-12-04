import tensorflow as tf
from tf.contrib.layers import conv2d, fully_connected


class AttentionDecoder():
  def __init__(self,
               rnn_cell,
               num_attention_units,
               attention_conv_kernel_size,
               is_training=True):
    self._rnn_cell = rnn_cell
    self._num_attention_units = num_attention_units
    self._attention_conv_kernel_size = attention_conv_kernel_size
    self._is_training = is_training

  def predict(self, feature_map, num_steps, decoder_inputs=None):
    if not self._is_training:
      raise RuntimeError('predict should only be called during training')
    if isinstance(feature_map, list):
      feature_map = feature_map[-1]

    batch_size = feature_map.get_shape()[0].value
    feature_map_size = tf.shape(feature_map_size)[1:3]

    initial_attention = tf.zeros(feature_map_size, tf.float32)
    initial_state = self._rnn_cell.zero_state(batch_size, tf.float32)
    initial_output = self._output_embedding_fn(tf.tile([symbols.GO], batch_size))

    outputs_list = []

    last_state = initial_state
    last_attention = initial_attention
    last_output = initial_output
    for i in range(num_steps):
      with tf.variable_scope('PredictStep_{}'.format(i), reuse=(i > 0)):
        output, new_state, attention_weights = \
          self._predict_step(
            feature_map,
            last_state,
            last_attention,
            last_output
          )
        outputs_list.append(output)
        last_state = new_state
        last_attention = attention_weights
        last_output = self._output_embedding_fn(decoder_inputs[:,i])
    outputs = tf.concat(outputs_list, axis=1) # => [batch_size, num_steps, output_dims]
    return outputs

  def _predict_step(self, feature_map, last_state, last_attention, last_output):
    """
    Args:
      feature_map: a float32 tensor with shape [batch_size, map_height, map_width, depth]
      last_state: a float32 tensor with shape [batch_size, ]
      last_attention: a float32 tensor with shape [batch_size, map_height, map_width, depth]
    """
    batch_size = feature_map.get_shape()[0].value
    feature_map_depth = feature_map.get_shape()[3].value
    if batch_size is None or feature_map_depth is None:
      raise ValueError('batch_size and feature_map_depth must be determined')

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
    ) # => [batch_size, map_height, map_width, num_attention_units]
    ws = fully_connected(
      last_state,
      self._num_attention_units,
      activation_fn=None,
      biases_initializer=tf.zeros_initializer(),
    ) # => [batch_size, num_attention_units], bias is added
    attention_sum = tf.bias_add(
      tf.add(vh, last_attention_conv),
      ws
    ) # => [batch_size, map_height, map_width, num_attention_units]
    attention_scores = slim.conv2d(
      tf.tanh(att_sum),
      1,
      activation_fn=None,
      biases_initializer=None,
    ) # => [batch_size, map_height, map_width, 1]
    attention_scores_flat = tf.reshape(
      tf.squeeze(attention_scores, axis=3),
      [batch_size, -1, 1]
    ) # => [batch_size, map_height * map_width, 1]
    attention_weights = tf.nn.softmax(
      attention_scores_flat, dim=1
    ) # => [batch_size, map_height * map_width, 1]
    feature_flat = tf.reshape(
      feature_map,
      [batch_size, -1, feature_map_depth]
    ) # => [batch_size, map_height * map_width, map_depth]
    feature_flat_trans = tf.transpose(
      feature_flat,
      [0, 2, 1]
    ) # => [batch_size, map_depth, map_height * map_width]
    glimpse = tf.squeeze(
      tf.matmul(feature_flat_trans, feature_flat),
      axis=2
    ) # [1, map_depth]

    output, new_state = self._rnn(glimpse, last_state)

    return output, new_state, attention_weights
