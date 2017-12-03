import tensorflow as tf



class AttentionRecognizer(core.RecognitionModel):

  def __init__(self,
               num_classes=None,
               feature_extractor=None):
    super(AttentionRecognizer, self).__init__(num_classes)
    self._feature_extractor = feature_extractor

  def predict(self, preprocessed_images):
    """
    Args:
      preprocessed_images: a float tensor with shape [batch_size, image_height, image_width, 3]
    Returns:
      predictions_dict: a diction of predicted tensors
    """
    with tf.variable_scope('FeatureExtractor'):
      feature_maps = self._feature_extractor.extract_features(preprocessed_images)

    with tf.variable_scope('Decoder'):
      chars_predictions = self._decoder.predict(feature_maps)

    predictions_dict = {
      'chars_predictions': chars_predictions
    }
    return predictions_dict

  def decode(self, predictions_dict):
    raise NotImplementedError('')

  def loss(self, predictions_dict):
    raise NotImplementedError('')
