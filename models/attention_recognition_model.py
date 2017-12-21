import tensorflow as tf
from tensorflow.contrib import seq2seq
from tensorflow.contrib.layers import conv2d

from rare.core import label_map
from rare.core import recognition_model, recognition_model_pb2
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
              start_label=None,
              end_label=None,
              scope=None):
    if isinstance(feature_map, list):
      feature_map = feature_map[-1]

    with tf.variable_scope(scope, 'Predict', [feature_map]):
      batch_size, _, _, map_depth = shape_utils.combined_static_and_dynamic_shape(feature_map)
      if batch_size is None or map_depth is None:
        raise ValueError('batch_size and map_depth must be static')

      embedding_fn = functools.partial(tf.one_hot, depth=num_classes)
      memory = conv2d(feature_map,
                      self._num_attention_units,
                      kernel_size=1,
                      stride=1,
                      scope='MemoryConv')
      memory = tf.reshape(memory, [batch_size, -1, self._num_attention_units])
      attention_mechanism = seq2seq.BahdanauAttention(
        self._num_attention_units,
        memory,
        memory_sequence_length=None) # all full lenghts
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
          start_tokens=tf.tile([start_label], [batch_size]),
          end_token=end_label)

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
    return outputs.rnn_output, outputs.sample_id, output_lengths


class AttentionRecognitionModel(recognition_model.RecognitionModel):

  def __init__(self,
               feature_extractor=None,
               birnn_cells_list=[],
               predictor=None,
               label_map=None,
               loss=None,
               is_training=True):
    self._feature_extractor = feature_extractor
    self._birnn_cells_list = birnn_cells_list
    self._predictor = predictor
    self._label_map = label_map
    self._loss = loss
    self._is_training = is_training
    self._groundtruth_dict = {}
    self.start_label = 0
    self.end_label = 1

  @property
  def num_classes(self):
    return self._label_map.num_classes + 2

  def preprocess(self, resized_inputs, scope=None):
    with tf.variable_scope(scope, 'ModelPreprocess', [resized_inputs]) as scope:
      if resized_inputs.dtype is not tf.float32:
        raise ValueError('`preprocess` expects a tf.float32 tensor')
      preprocessed_inputs = self._feature_extractor.preprocess(resized_inputs)
    return preprocessed_inputs

  def predict(self, preprocessed_images):
    """
    Args:
      preprocessed_images: a float tensor with shape [batch_size, image_height, image_width, 3]
    Returns:
      predictions_dict: a dictionary of predicted tensors
    """
    with tf.variable_scope('FeatureExtractor') as scope:
      batch_size = shape_utils.combined_static_and_dynamic_shape(preprocessed_images)[0]
      feature_maps = self._feature_extractor.extract_features(preprocessed_images, scope=scope)

      if len(self._birnn_cells_list) > 0:
        feature_map = feature_maps[0]
        batch_size, _, _, depth = shape_utils.combined_static_and_dynamic_shape(feature_map)  
        rnn_inputs = tf.reshape(feature_map, [batch_size, -1, depth])
        rnn_outputs = rnn_inputs
        for i, (fw_cell, bw_cell) in enumerate(self._birnn_cells_list):
          with tf.variable_scope('BidirectionalRnn_{}'.format(i)):
            (output_fw, output_bw), _ = tf.nn.bidirectional_dynamic_rnn(
              fw_cell, bw_cell, rnn_inputs, time_major=False, dtype=tf.float32)
            rnn_outputs = tf.concat([output_fw, output_bw], axis=2)
            rnn_inputs = rnn_outputs
        feature_maps = [tf.reshape(rnn_outputs, [batch_size, 1, -1, rnn_outputs.get_shape()[2].value])]

    with tf.variable_scope('Predictor') as scope:
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
        scope=scope
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
        self._groundtruth_dict['prediction_labels'],
        self._groundtruth_dict['prediction_labels_lengths']
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
        [start_labels, groundtruth_text_labels],
        axis=1)
      prediction_labels = tf.concat(
        [groundtruth_text_labels, end_labels],
        axis=1)
      self._groundtruth_dict['text_labels'] = groundtruth_text_labels
      self._groundtruth_dict['text_lengths'] = text_lengths
      self._groundtruth_dict['decoder_inputs'] = decoder_inputs
      self._groundtruth_dict['decoder_inputs_lengths'] = text_lengths + 1
      self._groundtruth_dict['prediction_labels'] = prediction_labels
      self._groundtruth_dict['prediction_labels_lengths'] = text_lengths + 1


def _build_attention_predictor(config, is_training):
  if not isinstance(config, recognition_model_pb2.AttentionPredictor):
    raise ValueError('config not of type recognition_model_pb2.AttentionPredictor')
  attention_predictor_oneof = model_config.WhichOneof('attention_predictor_oneof')

  if attention_predictor_oneof == 'bahdanau_attention_predictor':
    attention_predictor_config = config.bahdanau_attention_predictor

    rnn_cell_object = rnn_cell.build(attention_predictor_config.rnn_cell)
    attention_predictor_object = BahdanauAttentionPredictor(
      rnn_cell=rnn_cell_object,
      num_attention_units=attention_predictor_config.num_attention_units,
      max_num_steps=attention_predictor_config.max_num_steps,
      is_training=is_training
    )
    return attention_predictor_object
  else:
    raise ValueError('Unknown attention_predictor_oneof: {}'.format(attention_predictor_oneof))


def build(config, is_training):
  if not isinstance(config, recognition_model_pb2.AttentionRecognitionModel):
    raise ValueError('config not of type recognition_model_pb2.AttentionRecognitionModel')

  feature_extractor_object = feature_extractor.build(config.feature_extractor, is_training=is_training)
  bidirectional_rnn_cells_list = []
  for rnn_cell_config in model_config.bidirectional_rnn_cell:
    bidirectional_rnn_cells_list.append(
      (rnn_cell.build(rnn_cell_config),
       rnn_cell.build(rnn_cell_config))
    )
  attention_predictor_object = _build_attention_predictor(
    model_config.attention_predictor, is_training)
  label_map_object = label_map.build(model_config.label_map)
  loss_object = loss.build(model_config.loss)

  model_object = AttentionRecognitionModel(
    feature_extractor=feature_extractor_object,
    birnn_cells_list=bidirectional_rnn_cells_list,
    predictor=attention_predictor_object,
    label_map=label_map_object,
    loss=loss_object,
    is_training=is_training
  )
  return model_object
