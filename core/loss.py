import tensorflow as tf

from rare.utils import shape_utils


class SequenceCrossEntropyLoss(object):
  def __init__(self,
               sequence_normalize=None,
               sample_normalize=None,
               weight=None):
    self._sequence_normalize = sequence_normalize
    self._sample_normalize = sample_normalize
    self._weight = weight

  def __call__(self, logits, labels, lengths, scope=None):
    """
    Args:
      logits: float32 tensor with shape [batch_size, max_time, num_classes]
      labels: int32 tensor with shape [batch_size, max_time]
      lengths: int32 tensor with shape [batch_size]
    """
    with tf.name_scope(scope, 'SequenceCrossEntropyLoss', [logits, labels, lengths]):
      raw_losses = tf.nn.sparse_softmax_cross_entropy_with_logits(
        labels=labels,
        logits=logits
      )
      batch_size, max_time = shape_utils.combined_static_and_dynamic_shape(labels)
      mask = tf.less(
        tf.tile([tf.range(max_time)], [batch_size, 1]),
        tf.expand_dims(lengths, 1),
        name='mask'
      )
      masked_losses = tf.multiply(
        raw_losses,
        tf.cast(mask, tf.float32),
        name='masked_losses'
      ) # => [batch_size, max_time]
      row_losses = tf.reduce_sum(masked_losses, 1, name='row_losses')
      if self._sequence_normalize:
        loss = tf.truediv(
          row_losses,
          tf.cast(tf.maximum(lengths, 1), tf.float32),
          name='seq_normed_losses')
      loss = tf.reduce_sum(row_losses)
      if self._sample_normalize:
        loss = tf.truediv(
          loss,
          tf.cast(tf.maximum(batch_size, 1), tf.float32))
      if self._weight:
        loss = loss * self._weight
    return loss


class L2RegressionLoss(object):
  def __init__(self, weight=None):
    self._weight = weight

  def __call__(self, prediction, target, scope=None):
    with tf.name_scope(scope, 'L2RegressionLoss', [prediction, target]):
      losses = tf.nn.norm(prediction, target, ord=2, axis=1, keep_dims=False)
      loss = tf.reduce_mean(losses)
      if self._weight is not None:
        loss = loss * self._weight
    return loss
