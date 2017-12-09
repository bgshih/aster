import tensorflow as tf
from tensorflow.contrib.layers import conv2d, fully_connected

from rare.utils import shape_utils


class AttentionPredictor(object):
  def __init__(self,
               rnn_cell,
               num_attention_units,
               attention_conv_kernel_size,
               output_embedding=None,
               max_num_steps=None,
               is_training=True):
    self._rnn_cell = rnn_cell
    self._num_attention_units = num_attention_units
    self._attention_conv_kernel_size = attention_conv_kernel_size
    self._max_num_steps = max_num_steps
    self._output_embedding = output_embedding
    self._is_training = is_training

  def decode(self,
             feature_map,
             num_steps,
             num_classes=None,
             decoder_inputs=None):
    """Decode sequence output.
    Args:
      feature_map: a float32 tensor of shape [batch_size, height, width, depth]
      num_steps: a int32 scalar tensor indicating the number of decoding steps
      decoder_inputs: an int tensor of shape [batch_size, num_steps].
                      Should be groundtruth labels with GO padded to the front.
    """
    if not self._is_training:
      raise NotImplementedError('')
    if isinstance(feature_map, list):
      feature_map = feature_map[-1]
    
    with tf.variable_scope(None, 'Decode',
                           [feature_map, num_steps, decoder_inputs]):
      batch_size = shape_utils.combined_static_and_dynamic_shape(feature_map)[0]
      initial_state = self._rnn_cell.zero_state(batch_size, tf.float32)
      initial_alignment = tf.expand_dims(
        tf.zeros(tf.shape(feature_map)[:3], dtype=tf.float32),
        axis=3
      )
      rnn_outputs_list = []
      last_state = initial_state
      last_alignment = initial_alignment

      # project feature map to vh
      vh = conv2d(
        feature_map,
        self._num_attention_units,
        kernel_size=1,
        stride=1,
        biases_initializer=None,
        scope='Conv_vh'
      ) # => [batch_size, map_height, map_width, num_attention_units]

      for i in range(self._max_num_steps):
        if i > 0: tf.get_variable_scope().reuse_variables()

        with tf.name_scope('Step_{}'.format(i)):
          is_in_range = tf.less(i, num_steps)
          decoder_input_i = tf.cond(is_in_range,
            true_fn = lambda: self._output_embedding.embed(decoder_inputs[:,i], num_classes),
            false_fn = lambda: tf.zeros([batch_size, num_classes])
          )
          output_candidate, new_state, alignment = \
            self._decode_step(vh, last_state, last_alignment, decoder_input_i)
          output = tf.cond(is_in_range,
            true_fn = lambda: output_candidate,
            false_fn = lambda: tf.zeros_like([batch_size, self._rnn_cell.output_size])
          )
          rnn_outputs_list.append(output)
          last_state = new_state
          last_alignment = alignment

      rnn_outputs = tf.concat(rnn_outputs_list, axis=1)
      rnn_outputs = tf.slice(
        rnn_outputs,
        [0, 0, 0],
        [-1, num_steps, -1]
      ) # => [batch_size, num_steps, output_dims]

      logits = fully_connected(
        rnn_outputs,
        num_classes,
        activation_fn=None,
        scope='FullyConnected_logits'
      ) # => [batch_size, num_steps, num_classes]
    return logits

  def _decode_step(self, vh, last_state, last_attention, decoder_input):
    """
    Args:
      vh: a float32 tensor with shape [batch_size, map_height, map_width, num_attention_units]
      last_state: a float32 tensor with shape [batch_size, ]
      last_attention: a float32 tensor with shape [batch_size, map_height, map_width, depth]
      decoder_input: a float32 tensor with shape [batch_size, decoder_input_size]
    """
    batch_size, map_height, map_width, map_depth = \
      shape_utils.combined_static_and_dynamic_shape(feature_map)

    feature_map_depth = feature_map.get_shape()[3].value
    if batch_size is None or feature_map_depth is None:
      raise ValueError('batch_size and feature_map_depth must be determined')

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
    alignment_flat = tf.nn.softmax(
      attention_scores_flat, dim=1
    ) # => [batch_size, map_height * map_width, 1]
    alignment = tf.reshape(
      alignment_flat,
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
    return output, new_state, alignment
