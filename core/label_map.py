import string

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
  def go_label(self):
    return 1

  @property
  def unk_label(self):
    return 2

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
    chars = ['', '', ''] + self._character_set
    labels = [self.eos_label, self.go_label, self.unk_label] + \
             list(range(self.num_control_symbols, self.num_control_symbols + self.num_characters))
    char_to_label_table = tf.contrib.lookup.HashTable(
      tf.contrib.lookup.KeyValueTensorInitializer(
        chars[self.num_control_symbols:],
        labels[self.num_control_symbols:]
        key_dtype=tf.string,
        value_dtype=tf.int32
      )
    )
    label_to_char_table = tf.contrib.lookup.HashTable(
      tf.contrib.lookup.KeyValueTensorInitializer(
        labels,
        chars,
        key_dtype=tf.int32,
        value_dtype=tf.string
      )
    )
    return char_to_label_table, label_to_char_table

  def text_to_labels(self, text):
    """Convert text strings to label sequences.
    Args:
      text: ascii encoded string tensor of shape [batch_size]

    """
    batch_size = tf.shape(text)[0]
    chars = tf.string_split(strings, delimiter='')
    labels_sparse = self._char_to_label_table.lookup(chars.values)
    labels_dense = tf.sparse_tensor_to_dense(
      chars.indices,
      chars.dense_shape,
      labels_sparse,
      default_value=self.eos_label
    ) # => [batch_size, max_text_length]

    # pad eos symbols
    labels_padded = tf.concat(
      [labels_dense,
       tf.fill([batch_size, self.num_eos])],
      axis=1
    ) # => [batch_size, max_text_length + self.num_eos]

    def _convert(text):
      chars = tf.string_split(text, )
      self._char_to_label_table.lookup()


  def labels_to_text(self, labels):
    """Convert labels to text strings.
    Args:
      labels: int32 tensor []
    """
    pass
