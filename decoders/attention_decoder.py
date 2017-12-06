import tensorflow as tf
from tensorflow.contrib.layers import conv2d, fully_connected

from rare.utils import shape_utils


class AttentionDecoder():
  def __init__(self,
               rnn_cell,
               num_attention_units,
               attention_conv_kernel_size,
               output_embedding=None,
               is_training=True):
    self._rnn_cell = rnn_cell
    self._num_attention_units = num_attention_units
    self._attention_conv_kernel_size = attention_conv_kernel_size
    self._output_embedding = output_embedding
    self._is_training = is_training

  def decode(self,
             feature_map,
             num_steps,
             num_classes=None,
             decoder_inputs=None,
             scope=None):
    """Decode sequence output.
    Args:
      feature_map: a float32 tensor of shape [batch_size, height, width, depth]
      num_steps: a Python integer
      decoder_inputs: an int tensor of shape [batch_size, steps].
                      It should be groundtruth labels with GO and END symbols appended
                      during training and GO symbols during testing.
    """
    if not self._is_training:
      raise NotImplementedError('')
    if isinstance(feature_map, list):
      feature_map = feature_map[-1]
    
    with tf.variable_scope(scope, 'AttentionDecoder',
                           [feature_map, decoder_inputs]):
      batch_size = shape_utils.combined_static_and_dynamic_shape(feature_map)[0]
      initial_attention = tf.expand_dims(
        tf.zeros(tf.shape(feature_map)[:3], dtype=tf.float32),
        axis=3
      )
      initial_state = self._rnn_cell.zero_state(batch_size, tf.float32)
      rnn_outputs_list = []
      last_state = initial_state
      last_attention = initial_attention

      for i in range(num_steps):
        if i > 0: tf.get_variable_scope().reuse_variables()
        with tf.name_scope('Step_{}'.format(i)):
          decoder_input_i = self._output_embedding.embed(decoder_inputs[:,i], num_classes)
          output, new_state, attention_weights = \
            self._decode_step(
              feature_map,
              last_state,
              last_attention,
              decoder_input_i
            )
          rnn_outputs_list.append(output)
          last_state = new_state
          last_attention = attention_weights
      rnn_outputs = tf.concat(rnn_outputs_list, axis=1) # => [batch_size, num_steps, output_dims]

      logits = fully_connected(
        rnn_outputs,
        num_classes,
        activation_fn=None,
        scope='FullyConnected_logits'
      )
    return logits

  def _decode_step(self, feature_map, last_state, last_attention, decoder_input):
    """
    Args:
      feature_map: a float32 tensor with shape [batch_size, map_height, map_width, depth]
      last_state: a float32 tensor with shape [batch_size, ]
      last_attention: a float32 tensor with shape [batch_size, map_height, map_width, depth]
      decoder_input: a float32 tensor with shape [batch_size, decoder_input_size]
    """
    batch_size, map_height, map_width, map_depth = \
      shape_utils.combined_static_and_dynamic_shape(feature_map)

    feature_map_depth = feature_map.get_shape()[3].value
    if batch_size is None or feature_map_depth is None:
      raise ValueError('batch_size and feature_map_depth must be determined')

    vh = conv2d(
      feature_map,
      self._num_attention_units,
      kernel_size=1,
      stride=1,
      biases_initializer=None,
      scope='Conv_vh'
    ) # => [batch_size, map_height, map_width, num_attention_units]
    last_attention_conv = conv2d(
      last_attention,
      self._num_attention_units,
      kernel_size=self._attention_conv_kernel_size,
      stride=1,
      biases_initializer=None,
      scope='Conv_attention'
    ) # => [batch_size, map_height, map_width, num_attention_units]
    ws = tf.reshape(
      fully_connected(
        last_state,
        self._num_attention_units,
        activation_fn=None,
        biases_initializer=tf.zeros_initializer(),
        scope='FullyConnected_state'
      ),
      [batch_size, 1, 1, self._num_attention_units]
    ) # => [batch_size, 1, 1, num_attention_units], bias is added
    attention_sum = tf.add(
      tf.add(vh, last_attention_conv),
      ws
    ) # => [batch_size, map_height, map_width, num_attention_units]
    attention_scores = conv2d(
      tf.tanh(attention_sum),
      1,
      kernel_size=1,
      activation_fn=None,
      biases_initializer=None,
      scope='Conv_scores'
    ) # => [batch_size, map_height, map_width, 1]
    attention_scores_flat = tf.reshape(
      tf.squeeze(attention_scores, axis=3),
      [batch_size, -1, 1]
    ) # => [batch_size, map_height * map_width, 1]
    attention_weights_flat = tf.nn.softmax(
      attention_scores_flat, dim=1
    ) # => [batch_size, map_height * map_width, 1]
    attention_weights = tf.reshape(
      attention_weights_flat,
      [batch_size, map_height, map_width, map_depth]
    )
    feature_flat = tf.reshape(
      feature_map,
      [batch_size, map_width * map_height, map_depth]
    ) # => [batch_size, map_height * map_width, map_depth]
    feature_flat_trans = tf.transpose(
      feature_flat,
      [0, 2, 1]
    ) # => [batch_size, map_depth, map_height * map_width]
    glimpse = tf.squeeze(
      tf.matmul(feature_flat_trans, feature_flat),
      axis=2
    ) # [batch_size, map_depth]

    rnn_input = tf.concat([glimpse, decoder_input], axis=1)
    output, new_state = self._rnn_cell(rnn_input, last_state)
    return output, new_state, attention_weights
