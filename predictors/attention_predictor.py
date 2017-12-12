import functools

import tensorflow as tf
from tensorflow.contrib import seq2seq

from rare.utils import shape_utils


class BahdanauAttentionPredictor(object):
  """Attention decoder based on tf.contrib.seq2seq"""

  def __init__(self,
               rnn_cell,
               num_attention_units=None,
               max_num_steps=None,
               is_training=True):
    self._rnn_cell = rnn_cell
    self._num_attention_units = num_attention_units
    self._max_num_steps = max_num_steps
    self._is_training = is_training

  def predict(self,
              feature_map,
              decoder_inputs=None,
              decoder_inputs_lengths=None,
              num_classes=None,
              go_label=None,
              eos_label=None,
              scope=None):
    if isinstance(feature_map, list):
      feature_map = feature_map[-1]

    with tf.variable_scope(scope, 'Predict', [feature_map]):
      batch_size, _, _, map_depth = shape_utils.combined_static_and_dynamic_shape(feature_map)
      if batch_size is None or map_depth is None:
        raise ValueError('batch_size and map_depth must be static')

      embedding_fn = functools.partial(tf.one_hot, depth=num_classes)

      memory = tf.reshape(feature_map, [batch_size, -1, map_depth])
      attention_mechanism = seq2seq.BahdanauAttention(
        self._num_attention_units,
        memory,
        memory_sequence_length=None, # all full lenghts
      )
      attention_cell = seq2seq.AttentionWrapper(
        self._rnn_cell,
        attention_mechanism)

      if self._is_training:
        helper = seq2seq.TrainingHelper(
          embedding_fn(decoder_inputs),
          sequence_length=decoder_inputs_lengths,
          time_major=False)
      else:
        helper = seq2seq.GreedyEmbeddingHelper(
          embedding=embedding_fn,
          start_tokens=tf.tile([go_label], [batch_size]),
          end_token=eos_label)

      attention_decoder = seq2seq.BasicDecoder(
        cell=attention_cell,
        helper=helper,
        initial_state=attention_cell.zero_state(batch_size, tf.float32),
        output_layer=tf.layers.Dense(num_classes))
      outputs, _, output_lengths = seq2seq.dynamic_decode(
        decoder=attention_decoder,
        output_time_major=False,
        impute_finished=True,
        maximum_iterations=self._max_num_steps)

    output_logits = outputs.rnn_output
    output_labels = outputs.sample_id
    return output_logits, output_labels, output_lengths
