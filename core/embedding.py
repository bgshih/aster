import tensorflow as tf


class Embedding(object):
  def __init__(self):
    pass

  def embed(self, symbols):
    pass


class OneHotEmbedding(Embedding):

  def embed(self, labels, depth):
    """
    Args:
      labels: int32 tensor with shape [batch_size]
      depth: Python integer
    """
    return tf.one_hot(
      labels,
      depth,
      on_value=1.0,
      off_value=0.0,
      dtype=tf.float32
    )
