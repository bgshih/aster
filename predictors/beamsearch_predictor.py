import logging
import functools

import tensorflow as tf
from tensorflow.contrib import seq2seq


class BeamsearchPredictor(object):
  """Attention decoder based on tf.contrib.seq2seq"""

  def __init__(self,
               rnn_cell=None,
               rnn_regularizer=None,
               fc_hyperparams=None,
               num_attention_units=None,
               max_num_steps=None,
               multi_attention=False,
               is_training=True):
    self._rnn_cell = rnn_cell
    self._rnn_regularizer = rnn_regularizer
    self._fc_hyperparams = fc_hyperparams
    self._num_attention_units = num_attention_units
    self._max_num_steps = max_num_steps
    self._multi_attention = multi_attention
    self._is_training = is_training

  def predict(self,
              feature_maps,
              decoder_inputs=None,
              decoder_inputs_lengths=None,
              num_classes=None,
              start_label=None,
              end_label=None,
              scope=None):
    if not isinstance(feature_maps, (list, tuple)):
      raise ValueError('`feature_maps` must be list of tuple')

    with tf.variable_scope(scope, 'Predict', feature_maps):
      feature_sequences = [tf.squeeze(map, axis=1) for map in feature_maps]
      if self._multi_attention:
        attention_mechanism = []
        for i, feature_sequence in enumerate(feature_sequences):
          memory = feature_sequence
          if self._beam_width > 0:
            memory = seq2seq.tile_batch(memory, multiplier=self._beam_width)
          attention_mechanism.append(
            seq2seq.BahdanauAttention(
              self._num_attention_units,
              memory,
              memory_sequence_length=None
            )
          )
      else:
        memory = tf.concat(feature_sequences, axis=1)
        if self._beam_width > 0:
          memory = seq2seq.tile_batch(memory, multiplier=self._beam_width)
        attention_mechanism = seq2seq.BahdanauAttention(
          self._num_attention_units,
          memory,
          memory_sequence_length=None
        )

      attention_cell = seq2seq.AttentionWrapper(
        self._rnn_cell,
        attention_mechanism,
        output_attention=False)

      batch_size = shape_utils.combined_static_and_dynamic_shape(feature_maps[0])[0]
      embedding_fn = functools.partial(tf.one_hot, depth=num_classes)
      
      if self._is_training:
        helper = seq2seq.TrainingHelper(
          embedding_fn(decoder_inputs),
          sequence_length=decoder_inputs_lengths,
          time_major=False)
      else:
        helper = seq2seq.GreedyEmbeddingHelper(
          embedding=embedding_fn,
          start_tokens=tf.tile([start_label], [batch_size]),
          end_token=end_label)

      output_layer = tf.layers.Dense(
        num_classes,
        activation=None,
        use_bias=True,
        kernel_initializer=tf.variance_scaling_initializer(),
        bias_initializer=tf.zeros_initializer())
      attention_decoder = seq2seq.BasicDecoder(
        cell=attention_cell,
        helper=helper,
        initial_state=attention_cell.zero_state(batch_size, tf.float32),
        output_layer=output_layer)
      outputs, _, output_lengths = seq2seq.dynamic_decode(
        decoder=attention_decoder,
        output_time_major=False,
        impute_finished=False,
        maximum_iterations=self._max_num_steps)

      # apply regularizer
      filter_weights = lambda vars : [x for x in vars if x.op.name.endswith('kernel')]
      tf.contrib.layers.apply_regularization(
        self._rnn_regularizer,
        filter_weights(attention_cell.trainable_weights))

    return outputs.rnn_output, outputs.sample_id, output_lengths
