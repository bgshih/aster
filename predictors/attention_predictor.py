import logging
import functools

import tensorflow as tf
from tensorflow.contrib import seq2seq
from rare.core import predictor
from rare.core import sync_attention_wrapper
from rare.core import loss
from rare.utils import shape_utils
from rare.c_ops import ops


class AttentionPredictor(predictor.Predictor):
  """Attention decoder based on tf.contrib.seq2seq"""

  def __init__(self,
               rnn_cell=None,
               rnn_regularizer=None,
               num_attention_units=None,
               max_num_steps=None,
               multi_attention=False,
               beam_width=None,
               reverse=False,
               label_map=None,
               loss=None,
               sync=False,
               is_training=True):
    super(AttentionPredictor, self).__init__(is_training)
    self._rnn_cell = rnn_cell
    self._rnn_regularizer = rnn_regularizer
    self._num_attention_units = num_attention_units
    self._max_num_steps = max_num_steps
    self._multi_attention = multi_attention
    self._beam_width = beam_width
    self._reverse = reverse
    self._label_map = label_map
    self._sync = sync
    self._loss = loss

    if not self._is_training and not self._beam_width > 0:
      raise ValueError('Beam width must be > 0 during inference')

  @property
  def start_label(self):
    return 0

  @property
  def end_label(self):
    return 1

  @property
  def num_classes(self):
    return self._label_map.num_classes + 2

  def predict(self, feature_maps, scope=None):
    if not isinstance(feature_maps, (list, tuple)):
      raise ValueError('`feature_maps` must be list of tuple')

    with tf.variable_scope(scope, 'Predict', feature_maps):
      batch_size = shape_utils.combined_static_and_dynamic_shape(feature_maps[0])[0]      
      decoder_cell = self._build_decoder_cell(feature_maps)
      decoder = self._build_decoder(decoder_cell, batch_size)

      outputs, _, output_lengths = seq2seq.dynamic_decode(
        decoder=decoder,
        output_time_major=False,
        impute_finished=False,
        maximum_iterations=self._max_num_steps
      )
      # apply regularizer
      filter_weights = lambda vars : [x for x in vars if x.op.name.endswith('kernel')]
      tf.contrib.layers.apply_regularization(
        self._rnn_regularizer,
        filter_weights(decoder_cell.trainable_weights))

      outputs_dict = None
      if self._is_training:
        assert isinstance(outputs, seq2seq.BasicDecoderOutput)
        outputs_dict = {
          'labels': outputs.sample_id,
          'logits': outputs.rnn_output,
        }
      else:
        assert isinstance(outputs, seq2seq.FinalBeamSearchDecoderOutput)
        prediction_labels = outputs.beam_search_decoder_output.predicted_ids[:,:,0]
        prediction_lengths = output_lengths[:,0]
        prediction_scores = tf.gather_nd(
          outputs.beam_search_decoder_output.scores[:,:,0],
          tf.stack([tf.range(batch_size), prediction_lengths-1], axis=1)
        )
        outputs_dict = {
          'labels': prediction_labels,
          'scores': prediction_scores,
          'lengths': prediction_lengths
        }
    return outputs_dict

  def loss(self, predictions_dict, scope=None):
    assert 'logits' in predictions_dict
    with tf.variable_scope(scope, 'Loss', list(predictions_dict.values())):
      loss_tensor = self._loss(
        predictions_dict['logits'],
        self._groundtruth_dict['decoder_targets'],
        self._groundtruth_dict['decoder_lengths']
      )
    return loss_tensor

  def provide_groundtruth(self, groundtruth_text, scope=None):
    with tf.name_scope(scope, 'ProvideGroundtruth', [groundtruth_text]):
      batch_size = shape_utils.combined_static_and_dynamic_shape(groundtruth_text)[0]
      if self._reverse:
        groundtruth_text = ops.string_reverse(groundtruth_text)
      text_labels, text_lengths = self._label_map.text_to_labels(
        groundtruth_text,
        pad_value=self.end_label,
        return_lengths=True)
      start_labels = tf.fill([batch_size, 1], tf.constant(self.start_label, tf.int64))
      end_labels = tf.fill([batch_size, 1], tf.constant(self.end_label, tf.int64))
      if not self._sync:
        decoder_inputs = tf.concat([start_labels, start_labels, text_labels], axis=1)
        decoder_targets = tf.concat([start_labels, text_labels, end_labels], axis=1)
        decoder_lengths = text_lengths + 2
      else:
        decoder_inputs = tf.concat([start_labels, text_labels], axis=1)
        decoder_targets = tf.concat([text_labels, end_labels], axis=1)
        decoder_lengths = text_lengths + 2
      self._groundtruth_dict['decoder_inputs'] = decoder_inputs
      self._groundtruth_dict['decoder_targets'] = decoder_targets
      self._groundtruth_dict['decoder_lengths'] = decoder_lengths

  def postprocess(self, predictions_dict, scope=None):
    assert 'scores' in predictions_dict
    with tf.variable_scope(scope, 'Postprocess', list(predictions_dict.values())):
      text = self._label_map.labels_to_text(predictions_dict['labels'])
      if self._reverse:
        text = ops.string_reverse(text)
      scores = predictions_dict['scores']
    return {'text': text, 'scores': scores}

  def _build_decoder_cell(self, feature_maps):
    attention_mechanism = self._build_attention_mechanism(feature_maps)
    wrapper_class = seq2seq.AttentionWrapper if not self._sync else sync_attention_wrapper.SyncAttentionWrapper
    decoder_cell = wrapper_class(
      self._rnn_cell,
      attention_mechanism,
      output_attention=False)
    return decoder_cell

  def _build_attention_mechanism(self, feature_maps):
    """Build (possibly multiple) attention mechanisms."""
    def _build_single_attention_mechanism(memory):
      if not self._is_training:
        memory = seq2seq.tile_batch(memory, multiplier=self._beam_width)
      return seq2seq.BahdanauAttention(
        self._num_attention_units,
        memory,
        memory_sequence_length=None
      )
    
    feature_sequences = [tf.squeeze(map, axis=1) for map in feature_maps]
    if self._multi_attention:
      attention_mechanism = []
      for i, feature_sequence in enumerate(feature_sequences):
        memory = feature_sequence
        attention_mechanism.append(_build_single_attention_mechanism(memory))
    else:
      memory = tf.concat(feature_sequences, axis=1)
      attention_mechanism = _build_single_attention_mechanism(memory)
    return attention_mechanism

  def _build_decoder(self, decoder_cell, batch_size):
    embedding_fn = functools.partial(tf.one_hot, depth=self.num_classes)
    output_layer = tf.layers.Dense(
      self.num_classes,
      activation=None,
      use_bias=True,
      kernel_initializer=tf.variance_scaling_initializer(),
      bias_initializer=tf.zeros_initializer())
    if self._is_training:
      train_helper = seq2seq.TrainingHelper(
        embedding_fn(self._groundtruth_dict['decoder_inputs']),
        sequence_length=self._groundtruth_dict['decoder_lengths'],
        time_major=False)
      decoder = seq2seq.BasicDecoder(
        cell=decoder_cell,
        helper=train_helper,
        initial_state=decoder_cell.zero_state(batch_size, tf.float32),
        output_layer=output_layer)
    else:
      decoder = seq2seq.BeamSearchDecoder(
        cell=decoder_cell,
        embedding=embedding_fn,
        start_tokens=tf.tile([self.start_label], [batch_size * self._beam_width]),
        end_token=self.end_label,
        initial_state=decoder_cell.zero_state(batch_size * self._beam_width, tf.float32),
        beam_width=self._beam_width,
        output_layer=output_layer,
        length_penalty_weight=0.0)
    return decoder
