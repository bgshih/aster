import tensorflow as tf

from rare.core import label_map
from rare.utils import shape_utils


class AttentionRecognitionModel(object):

  def __init__(self,
               feature_extractor=None,
               label_map=None,
               loss=None):
    self._feature_extractor = feature_extractor
    self._label_map = label_map
    self._loss = loss
    self._groundtruth_dict = {}
    self._num_classes = label_map.num_labels

  @property
  def num_classes(self):
    return self._num_classes

  def preprocess(self, resized_inputs):
    if resized_inputs.dtype is not tf.float32:
      raise ValueError('`preprocess` expects a tf.float32 tensor')
    with tf.name_scope('Preprocess'):
      return self._feature_extractor.preprocess(resized_inputs)

  def predict(self, preprocessed_images, max_length):
    """
    Args:
      preprocessed_images: a float tensor with shape [batch_size, image_height, image_width, 3]
    Returns:
      predictions_dict: a diction of predicted tensors
    """
    if not self._is_training:
      raise ValueError('`predict` should only be called when self._is_training is True')

    with tf.variable_scope('FeatureExtractor'):
      feature_maps = self._feature_extractor.extract_features(preprocessed_images)

    with tf.variable_scope('Decoder'):
      groundtruth_labels = self._label_map.text_to_labels(
        self._groundtruth_dict['groundtruth_text']
      )
      batch_size = shape_utils.combined_static_and_dynamic_shape(groundtruth_labels)[0]
      go_labels = tf.fill([batch_size], tf.constant(self._label_map.go_label, dtype=tf.int64))
      decoder_inputs = tf.concat([go_labels, groundtruth_labels], axis=1)

      logits = self._decoder.predict(
        feature_maps,
        max_length,
        self._num_classes,
        decoder_inputs
      )

    predictions_dict = {
      'logits': logits
    }
    return predictions_dict

  def loss(self, predictions_dict):
    prediction_logits = predictions_dict['logits']
    groundtruth_labels = self._groundtruth_dict['padded_groundtruth_labels']
    groundtruth_lengths = self._groundtruth_dict['padded_groundtruth_lengths']
    return self._loss(prediction_logits, groundtruth_labels, groundtruth_lengths)

  def provide_groundtruth(self, groundtruth_text_list):
    self._groundtruth_dict['groundtruth_text'] = tf.stack(groundtruth_text_list, axis=0)
