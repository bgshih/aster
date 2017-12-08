import tensorflow as tf

from rare.core import label_mapping
from rare.utils import shape_utils


class AttentionRecognitionModel(core.RecognitionModel):

  def __init__(self,
               num_classes=None,
               feature_extractor=None,
               label_map=None,
               loss=None):
    super(AttentionRecognizer, self).__init__(num_classes)
    self._feature_extractor = feature_extractor
    self._label_map = label_map
    self._loss = loss
    self._groundtruth_dict = {}

  def predict(self, preprocessed_images, max_length):
    """
    Args:
      preprocessed_images: a float tensor with shape [batch_size, image_height, image_width, 3]
    Returns:
      predictions_dict: a diction of predicted tensors
    """
    with tf.variable_scope('FeatureExtractor'):
      feature_maps = self._feature_extractor.extract_features(preprocessed_images)

    with tf.variable_scope('Decoder'):
      groundtruth_labels = self._groundtruth_dict['padded_groundtruth_labels'] # => [batch_size, max_time]
      batch_size = shape_utils.combined_static_and_dynamic_shape(groundtruth_labels)[0]
      go_labels = tf.tile([label_mapping.GO_LABEL], batch_size)
      decoder_inputs = tf.concat([go_labels, groundtruth_labels], axis=1)
      logits = self._decoder.predict(feature_maps, max_length, num_classes, decoder_inputs)

    predictions_dict = {
      'logits': logits
    }
    return predictions_dict

  def loss(self, predictions_dict):
    prediction_logits = predictions_dict['logits']
    groundtruth_labels = self._groundtruth_dict['padded_groundtruth_labels']
    groundtruth_lengths = self._groundtruth_dict['padded_groundtruth_lengths']
    return self._loss(prediction_logits, groundtruth_labels, groundtruth_lengths)

  def provide_groundtruth(self, groundtruth_dict):
    if 'padded_groundtruth_labels' not in groundtruth_dict or \
       'padded_groundtruth_lengths' not in groundtruth_dict:
      raise ValueError('groundtruth_dict does not have all the expected keys')

    # groundtruth labels padded with EOS symbols
    self._groundtruth_dict['padded_groundtruth_labels'] = groundtruth_dict['padded_groundtruth_labels']
    # length of groundtruth labels, including padded EOS symbols
    self._groundtruth_dict['padded_groundtruth_lengths'] = groundtruth_dict['padded_groundtruth_lengths']

  def provide_groundtruth(self, groundtruth_transcript):
    pass
