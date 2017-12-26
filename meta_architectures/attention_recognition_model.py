import logging
import functools

import tensorflow as tf
from tensorflow.contrib import seq2seq

from rare.core import model
from rare.utils import shape_utils


class BahdanauAttentionPredictor(object):
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
          attention_mechanism.append(
            seq2seq.BahdanauAttention(
              self._num_attention_units,
              memory,
              memory_sequence_length=None
            )
          )
      else:
        memory = tf.concat(feature_sequences, axis=1)
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


class AttentionRecognitionModel(model.Model):

  def __init__(self,
               feature_extractor=None,
               predictor=None,
               label_map=None,
               loss=None,
               is_training=True):
    super(AttentionRecognitionModel, self).__init__(
      feature_extractor,
      is_training)
    self._predictor = predictor
    self._label_map = label_map
    self._loss = loss

    logging.info('Number of classes: {}'.format(self.num_classes))

  @property
  def start_label(self):
    return 0

  @property
  def end_label(self):
    return 1

  @property
  def num_classes(self):
    return self._label_map.num_classes + 2

  def predict(self, preprocessed_inputs, scope=None):
    with tf.variable_scope(scope, 'FeatureExtractor', [preprocessed_inputs]) as feat_scope:
      feature_maps = self._feature_extractor.extract_features(preprocessed_inputs, scope=feat_scope)
      
    with tf.variable_scope('Predictor') as predictor_scope:
      if self._is_training:
        decoder_inputs = self._groundtruth_dict['decoder_inputs']
        decoder_inputs_lengths = self._groundtruth_dict['decoder_inputs_lengths']
      else:
        decoder_inputs = None
        decoder_inputs_lengths = None

      logits, labels, lengths = self._predictor.predict(
        feature_maps,
        decoder_inputs=decoder_inputs,
        decoder_inputs_lengths=decoder_inputs_lengths,
        num_classes=self.num_classes,
        start_label=self.start_label,
        end_label=self.end_label,
        scope=predictor_scope
      )

    predictions_dict = {
      'logits': logits,
      'labels': labels,
      'lengths': lengths
    }
    return predictions_dict

  def loss(self, predictions_dict, scope=None):
    with tf.variable_scope(scope, 'Loss', list(predictions_dict.values())):
      loss = self._loss(
        predictions_dict['logits'],
        self._groundtruth_dict['target_labels'],
        self._groundtruth_dict['target_labels_lengths']
      )
    return {'RecognitionLoss': loss}

  def postprocess(self, predictions_dict, scope=None):
    with tf.variable_scope(scope, 'Postprocess', list(predictions_dict.values())):
      text = self._label_map.labels_to_text(predictions_dict['labels'])
    return {'text': text}

  def provide_groundtruth(self, groundtruth_text_list, scope=None):
    with tf.variable_scope(scope, 'ProvideGroundtruth', [groundtruth_text_list]):
      batch_size = len(groundtruth_text_list)
      groundtruth_text = tf.stack(groundtruth_text_list, axis=0)
      groundtruth_text_labels, text_lengths = self._label_map.text_to_labels(
        groundtruth_text,
        pad_value=self.end_label,
        return_lengths=True)
      start_labels = tf.fill([batch_size, 1], tf.constant(self.start_label, tf.int64))
      end_labels = tf.fill([batch_size, 1], tf.constant(self.end_label, tf.int64))
      decoder_inputs = tf.concat(
        [start_labels, start_labels, groundtruth_text_labels],
        axis=1)
      target_labels = tf.concat(
        [start_labels, groundtruth_text_labels, end_labels],
        axis=1)
      self._groundtruth_dict['text_labels'] = groundtruth_text_labels
      self._groundtruth_dict['text_lengths'] = text_lengths
      self._groundtruth_dict['decoder_inputs'] = decoder_inputs
      self._groundtruth_dict['decoder_inputs_lengths'] = text_lengths + 2
      self._groundtruth_dict['target_labels'] = target_labels
      self._groundtruth_dict['target_labels_lengths'] = text_lengths + 2
