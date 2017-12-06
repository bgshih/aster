import tensorflow as tf

from rare.utils import shape_utils


class SequenceCrossEntropyLoss(object):
  def __init__(self, sequence_normalize=True, sample_normalize=True):
    self._sequence_normalize = sequence_normalize
    self._sample_normalize = sample_normalize

  def __call__(self, logits, labels, lengths):
    """
    Args:
      logits: float32 tensor with shape [batch_size, max_time, num_classes]
      labels: int32 tensor with shape [batch_size, max_time]
      lengths: int32 tensor with shape [batch_size]
    """
    raw_losses = tf.nn.sparse_softmax_cross_entropy_with_logits(
      labels=labels,
      logits=logits
    )
    batch_size, max_time = shape_utils.combined_static_and_dynamic_shape(labels)
    mask = tf.less(
      tf.tile([tf.range(max_time)], [batch_size, 1]),
      tf.expand_dims(lengths, 1)
    )
    masked_losses = tf.multiply(
      raw_losses,
      tf.cast(mask, tf.float32)
    ) # => [batch_size, max_time]
    row_losses = tf.reduce_sum(masked_losses, 1)
    if self._sequence_normalize:
      row_losses = tf.truediv(row_losses, tf.cast(lengths, tf.float32))
    loss = tf.reduce_sum(row_losses)
    if self._sample_normalize:
      loss = tf.truediv(loss, tf.cast(batch_size, tf.float32))
    return loss
