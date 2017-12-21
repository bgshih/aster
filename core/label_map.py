import string
import logging

import tensorflow as tf

from rare.core import label_map_pb2
from rare.utils import shape_utils


class LabelMap(object):

  def __init__(self, config):
    if not isinstance(config, label_map_pb2.LabelMap):
      raise ValueError('config not of type label_map_pb2.LabelMap')
    self._character_set = _build_character_set(config.character_set)
    self._label_offset = config.label_offset
    self._unk_label = config.unk_label or config.label_offset
    logging.info('UNK label is {}'.format(self._unk_label))

    self._char_to_label_table, self._label_to_char_table = self._build_lookup_tables()  

  @property
  def num_classes(self):
    return len(self._character_set)

  def _build_lookup_tables(self):
    chars = self._character_set
    labels = list(range(self._label_offset, self._label_offset + self.num_classes))
    # Note: TF 1.4 does not support int32 keys
    char_to_label_table = tf.contrib.lookup.HashTable(
      tf.contrib.lookup.KeyValueTensorInitializer(
        chars, labels, key_dtype=tf.string, value_dtype=tf.int64),
      default_value=self._unk_label
    )
    label_to_char_table = tf.contrib.lookup.HashTable(
      tf.contrib.lookup.KeyValueTensorInitializer(
        labels, chars, key_dtype=tf.int64, value_dtype=tf.string),
      default_value=""
    )
    return char_to_label_table, label_to_char_table

  def text_to_labels(self,
                     text,
                     return_dense=True,
                     pad_value=-1,
                     return_lengths=False):
    """Convert text strings to label sequences.
    Args:
      text: ascii encoded string tensor with shape [batch_size]
      dense: whether to return dense labels
      pad_value: Value used to pad labels to the same length.
      return_lengths: if True, also return text lengths
    Returns:
      labels: sparse or dense tensor of labels
    """
    batch_size = tf.shape(text)[0]
    chars = tf.string_split(text, delimiter='')

    labels_sp = tf.SparseTensor(
      chars.indices,
      self._char_to_label_table.lookup(chars.values),
      chars.dense_shape
    )

    if return_dense:
      labels = tf.sparse_tensor_to_dense(labels_sp, default_value=pad_value)
    else:
      labels = labels_sp

    if return_lengths:
      text_lengths = tf.sparse_reduce_sum(
        tf.SparseTensor(
          chars.indices,
          tf.fill([tf.shape(chars.indices)[0]], 1),
          chars.dense_shape
        ),
        axis=1
      )
      text_lengths.set_shape([None])
      return labels, text_lengths
    else:
      return labels

  def labels_to_text(self, labels):
    """Convert labels to text strings.
    Args:
      labels: int32 tensor with shape [batch_size, max_label_length]
    Returns:
      text: string tensor with shape [batch_size]
    """
    if labels.dtype == tf.int32 or labels.dtype == tf.int64:
      labels = tf.cast(labels, tf.int64)
    else:
      raise ValueError('Wrong dtype of labels: {}'.format(labels.dtype))
    chars = self._label_to_char_table.lookup(labels)
    text = tf.reduce_join(chars, axis=1)
    return text


def _build_character_set(config):
  if not isinstance(config, label_map_pb2.CharacterSet):
    raise ValueError('config not of type label_map_pb2.CharacterSet')

  source_oneof = config.WhichOneof('source_oneof')
  character_set_string = None
  if source_oneof == 'text_file':
    file_path = config.text_file
    with open(file_path, 'r') as f:
      character_set_string = f.read()
  elif source_oneof == 'text_string':
    character_set_string = config.text_string

  if not config.delimiter:
    character_set = list(character_set_string)
  else:
    character_set = character_set_string.split()
  return character_set


def build(config):
  return LabelMap(config)
