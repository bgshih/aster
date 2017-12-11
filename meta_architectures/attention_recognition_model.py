import tensorflow as tf

from rare.core import label_map
from rare.utils import shape_utils


class AttentionRecognitionModel(object):

  def __init__(self,
               feature_extractor=None,
               predictor=None,
               label_map=None,
               loss=None):
    self._feature_extractor = feature_extractor
    self._predictor = predictor
    self._label_map = label_map
    self._loss = loss
    self._groundtruth_dict = {}

  @property
  def num_classes(self):
    return self._label_map.num_classes

  def preprocess(self, resized_inputs):
    if resized_inputs.dtype is not tf.float32:
      raise ValueError('`preprocess` expects a tf.float32 tensor')
    return self._feature_extractor.preprocess(resized_inputs)

  def predict(self, preprocessed_images):
    """
    Args:
      preprocessed_images: a float tensor with shape [batch_size, image_height, image_width, 3]
    Returns:
      predictions_dict: a diction of predicted tensors
    """
    batch_size = shape_utils.combined_static_and_dynamic_shape(preprocessed_images)[0]

    with tf.variable_scope('FeatureExtractor') as scope:
      feature_maps = self._feature_extractor.extract_features(preprocessed_images, scope=scope)

    with tf.variable_scope('Predictor') as scope:
      logits, labels, lengths = self._predictor.predict(
          feature_maps,
          decoder_inputs=self._groundtruth_dict['text_labels'],
          decoder_inputs_lengths=self._groundtruth_dict['text_lengths'],
          num_classes=self.num_classes,
          go_label=self._label_map.go_label,
          eos_label=self._label_map.eos_label,
          scope=scope)

    predictions_dict = {
      'logits': logits,
      'labels': labels,
      'lengths': lengths
    }
    return predictions_dict

  def loss(self, predictions_dict):
    loss = self._loss(
      predictions_dict['logits'],
      self._groundtruth_dict['text_labels'],
      self._groundtruth_dict['text_lengths']
    )
    return {'RecognitionLoss': loss}

  def provide_groundtruth(self, groundtruth_text_list):
    groundtruth_text = tf.stack(groundtruth_text_list, axis=0)
    groundtruth_labels, text_lengths = self._label_map.text_to_labels(
      groundtruth_text,
      pad_value=self._label_map.eos_label,
      return_lengths=True)
    self._groundtruth_dict['text_labels'] = groundtruth_labels
    self._groundtruth_dict['text_lengths'] = text_lengths
