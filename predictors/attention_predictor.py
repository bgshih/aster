from functools import partial
import tensorflow as tf
# from tensorflow.contrib.layers import conv2d, fully_connected

from rare.utils import shape_utils


class AttentionPredictor(object):
  def __init__(self,
               rnn_cell=None,
               label_map=None,
               is_training=True):
    self._rnn_cell = rnn_cell
    self._label_map = label_map
    self._is_training = is_training

  def predict(self,
              feature_map,
              decoder_inputs,
              decoder_inputs_lengths):
    num_classes = self._label_map.num_classes

    # feature_map: [batch_size, height, width, depth]
    feature_map_shape = shape_utils.combined_static_and_dynamic_shape(feature_map)
    batch_size, depth = feature_map_shape[0], feature_map_shape[3]

    if depth is None:
      raise ValueError('The depth of feature_map must be static')
    num_attention_units = depth

    attention_states = tf.reshape(feature_map, [batch_size, -1, depth])
    attention_mechanism = tf.contrib.seq2seq.LuongAttention(
      num_attention_units,
      attention_states,
      memory_sequence_length=None
    )
    decoder_cell = tf.contrib.seq2seq.AttentionWrapper(
      self._rnn_cell,
      attention_mechanism,
      attention_layer_size=num_units
    )

    if self._is_training:
      helper = tf.contrib.seq2seq.TrainingHelper(
        input=decoder_inputs,
        sequence_length=decoder_inputs_lengths
      )
    else:
      helper = tf.contrib.seq2seq.GreedyEmbeddingHelper(
        embedding=partial(tf.nn.one_hot, depth=self._label_map.num_classes),
        start_tokens=tf.tile([label_map.go_label], [batch_size]),
        end_token=label_map.eos_label
      )

    decoder = tf.contrib.seq2seq.BasicDecoder(
      cell=self._rnn_cell,
      helper=helper,
      initial_state=cell.zero_state(batch_size, tf.float32)
    )
    outputs, _, outputs_lengths = tf.contrib.seq2seq.dynamic_decode(
      decoder=decoder,
      output_time_major=False,
      impute_finished=False,
      maximum_iterations=30,
      parallel_iterations=32
    )

    return outputs
