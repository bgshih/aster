import tensorflow as tf
from tensorflow.contrib.layers import fully_connected



from rare.core import label_map
from rare.utils import shape_utils


class CtcRecognitionModel(object):

  def __init__(self,
               feature_extractor=None,
               bidirectional_rnn_cell_list=[],
               label_map=None,
               loss=None,
               is_training=True):
    self._feature_extractor = feature_extractor
    self._bidirectional_rnn_cell_list = bidirectional_rnn_cell_list
    self._label_map = label_map
    self._loss = loss
    self._is_training = is_training
    self._groundtruth_dict = {}

  @property
  def num_classes(self):
    # in tf.nn.ctc_loss, the largest label value is reserved for blank label
    return self._label_map.num_classes + 1

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
    with tf.variable_scope('FeatureExtractor') as scope:
      feature_map = self._feature_extractor.extract_features(preprocessed_images, scope=scope)[0]

    with tf.variable_scope('Predictor') as scope:
      feature_map_shape = shape_utils.combined_static_and_dynamic_shape(feature_map)
      batch_size, map_depth = feature_map_shape[0], feature_map_shape[3]
      if batch_size is None or map_depth is None:
        raise ValueError('batch_size and map_depth must be static')
      feature_sequence = tf.reshape(feature_map, [batch_size, -1, map_depth])
      # feature_sequence_list = tf.unstack(feature_sequence, axis=1)

      # build stacked bidirectional RNNs
      rnn_inputs = feature_sequence
      rnn_outputs = rnn_inputs
      for i, rnn_cell in enumerate(self._bidirectional_rnn_cell_list):
        with tf.variable_scope('BidirectionalRnn_{}'.format(i)):
          (output_fw, output_bw), _ = tf.nn.bidirectional_dynamic_rnn(
            rnn_cell, rnn_cell, rnn_inputs, time_major=False, dtype=tf.float32)
          rnn_outputs = tf.concat([output_fw, output_bw], axis=2)
          rnn_inputs = rnn_outputs

      logits = fully_connected(rnn_outputs, self.num_classes, activation_fn=None)
      return {'logits': logits}

  def loss(self, predictions_dict):
    logits = predictions_dict['logits']
    batch_size, max_time, _ = shape_utils.combined_static_and_dynamic_shape(logits)

    loss = tf.nn.ctc_loss(
      tf.cast(self._groundtruth_dict['text_labels_sparse'], tf.int32),
      predictions_dict['logits'],
      tf.fill([batch_size], max_time),
      time_major=False)
    # loss = tf.Print(
    #   loss,
    #   [tf.sparse_tensor_to_dense(self._groundtruth_dict['text_labels_sparse'], default_value=-1),
    #    tf.shape(predictions_dict['logits']),
    #    self._groundtruth_dict['text_lengths']],
    #   first_n=10,
    #   summarize=100)

    return {'RecognitionLoss': loss}

  def postprocess(self, predictions_dict):
    logits_time_major = tf.transpose(logits, [1,0,2])
    sparse_labels, log_prob = tf.nn.ctc_beam_search_decoder(
      predictions_dict['logits'],
      tf.fill([batch_size], tf.shape(logits_time_major)[0]),
      beam_width=10,
      top_paths=1,
    )
    labels = tf.sparse_tensor_to_dense(sparse_labels, default_value=-1)
    text = self._label_map.labels_to_text(labels)
    recognitions_dict = {
      'text': text
    }
    return recognitions_dict

  def provide_groundtruth(self, groundtruth_text_list):
    groundtruth_text = tf.stack(groundtruth_text_list, axis=0)
    groundtruth_text_labels_sp, text_lengths = self._label_map.text_to_labels(
      groundtruth_text,
      return_dense=False,
      return_lengths=True)
    self._groundtruth_dict['text_labels_sparse'] = groundtruth_text_labels_sp
    self._groundtruth_dict['text_lengths'] = text_lengths
