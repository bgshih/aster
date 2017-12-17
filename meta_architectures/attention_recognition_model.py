import tensorflow as tf

from rare.core import label_map
from rare.utils import shape_utils


class AttentionRecognitionModel(object):

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

  def preprocess(self, resized_inputs):
    if resized_inputs.dtype is not tf.float32:
      raise ValueError('`preprocess` expects a tf.float32 tensor')
    return self._feature_extractor.preprocess(resized_inputs)

  def predict(self, preprocessed_images):
    """
    Args:
      preprocessed_images: a float tensor with shape [batch_size, image_height, image_width, 3]
    Returns:
      predictions_dict: a dictionary of predicted tensors
    """
    batch_size = shape_utils.combined_static_and_dynamic_shape(preprocessed_images)[0]

    with tf.variable_scope('FeatureExtractor') as scope:
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

  def loss(self, predictions_dict):
    loss = self._loss(
      predictions_dict['logits'],
      self._groundtruth_dict['prediction_labels'],
      self._groundtruth_dict['prediction_labels_lengths']
    )
    return {'RecognitionLoss': loss}

  def postprocess(self, predictions_dict):
    text = self._label_map.labels_to_text(predictions_dict['labels'])
    recognitions_dict = {
      'text': text
    }
    return recognitions_dict

  def provide_groundtruth(self, groundtruth_text_list):
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
