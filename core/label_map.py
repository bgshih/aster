import string

import tensorflow as tf


class LabelMap(object):

  def __init__(self,
               character_set=None,
               num_eos=1):
    """
    Args:
      character_set: a list of utf8-encoded characters
      num_eos: number of end-of-sequence symbols to append
    """
    if not isinstance(character_set, list):
      raise ValueError('character_set must be provided as a list')
    if len(frozenset(character_set)) != len(character_set):
      raise ValueError('Found duplicate characters in character_set')

    self._character_set = character_set
    self._num_eos = num_eos
    (self._char_to_label_table, self._label_to_char_table) = self._build_lookup_tables()

  @property
  def eos_label(self):
    return 0

  @property
  def eos_char(self):
    return ''

  @property
  def go_label(self):
    return 1

  @property
  def go_char(self):
    return ''

  @property
  def unk_label(self):
    return 2

  @property
  def unk_char(self):
    return ''

  @property
  def num_control_symbols(self):
    return 3

  @property
  def num_characters(self):
    return len(self._character_set)

  @property
  def num_labels(self):
    return self.num_characters + self.num_control_symbols

  def _build_lookup_tables(self):
    chars = [self.eos_char, self.go_char, self.unk_char] + self._character_set
    labels = [self.eos_label, self.go_label, self.unk_label] + \
             list(range(self.num_control_symbols, self.num_control_symbols + self.num_characters))
    char_to_label_table = tf.contrib.lookup.HashTable(
      tf.contrib.lookup.KeyValueTensorInitializer(
        chars[self.num_control_symbols:],
        labels[self.num_control_symbols:],
        key_dtype=tf.string,
        value_dtype=tf.int64
      ),
      self.unk_label
    )
    label_to_char_table = tf.contrib.lookup.HashTable(
      tf.contrib.lookup.KeyValueTensorInitializer(
        labels,
        chars,
        key_dtype=tf.int64,
        value_dtype=tf.string
      ),
      self.unk_char
    )
    return char_to_label_table, label_to_char_table

  def text_to_labels(self, text):
    """Convert text strings to label sequences.
    Args:
      text: ascii encoded string tensor with shape [batch_size]
    Returns:
      labels_padded: labels padded with eos symbols
    """
    batch_size = tf.shape(text)[0]
    chars = tf.string_split(text, delimiter='')
    labels_values = self._char_to_label_table.lookup(chars.values)
    labels_dense = tf.sparse_to_dense(
      chars.indices,
      chars.dense_shape,
      labels_values,
      default_value=self.eos_label # use EOS to pad shorter text
    ) # => [batch_size, max_text_length]

    # pad eos symbols
    eos_labels = tf.fill(
      [batch_size, self._num_eos],
      tf.constant(self.eos_label, dtype=tf.int64)
    )
    labels_padded = tf.concat(
      [labels_dense, eos_labels],
      axis=1
    ) # => [batch_size, max_text_length + self._num_eos]
    return labels_padded

  def labels_to_text(self, labels):
    """Convert labels to text strings.
    Args:
      labels: int64 tensor with shape [batch_size, max_label_length]
    Returns:
      text: string tensor with shape [batch_size]
    """
    chars = self._label_to_char_table.lookup(labels)
    text = tf.reduce_join(chars, axis=1)
    return text
