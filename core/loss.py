import tensorflow as tf

from rare.utils import shape_utils
from rare.core import loss_pb2


class SequenceCrossEntropyLoss(object):
  def __init__(self, sequence_normalize=False, sample_normalize=True):
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
      row_losses = tf.truediv(
        row_losses,
        tf.cast(tf.maximum(lengths, 1), tf.float32))
    loss = tf.reduce_sum(row_losses)
    if self._sample_normalize:
      loss = tf.truediv(
        loss,
        tf.cast(tf.maximum(batch_size, 1), tf.float32))
    return loss


def build(config):
  if not isinstance(config, loss_pb2.Loss):
    raise ValueError('config not of type loss_pb2.Loss')
  loss_oneof = config.WhichOneof('loss_oneof')
  if loss_oneof == 'sequence_cross_entropy_loss':
    sequence_cross_entropy_loss_config = config.sequence_cross_entropy_loss
    return SequenceCrossEntropyLoss(
      sequence_normalize=sequence_cross_entropy_loss_config.sequence_normalize,
      sample_normalize=sequence_cross_entropy_loss_config.sample_normalize
    )
  else:
    raise ValueError('Unknown loss_oneof: {}'.format(loss_oneof))
