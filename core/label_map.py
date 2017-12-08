import string

class LabelMap(object):

  def __init__(self,
               case_sensitive=False,
               include_punctuations=False,
               num_eos=1):
    self._case_sensitive = case_sensitive
    self._include_punctuations = include_punctuations
    self._num_eos = num_eos

    char_to_label_map, label_to_char_map = \
      self._build_char_label_maps()

    char_to_label_map_items = char_to_label_map.items()
    self._char_to_label_table = tf.contrib.lookup.HashTable(
      tf.contrib.lookup.KeyValueTensorInitializer(
        keys=[t[0] for t in char_to_label_map_items],
        values=[t[1] for t in char_to_label_map_items],
      )
    )

    label_to_char_map_items = label_to_char_map.items()
    self._label_to_char_table = tf.contrib.lookup.HashTable(
      tf.contrib.lookup.KeyValueTensorInitializer(
        keys=[t[0] for t in label_to_char_map_items],
        values=[t[1] for t in label_to_char_map_items],
      )
    )

  @property
  def go_label(self):
    pass

  def _build_char_label_maps(self):
    char_to_label_map = {}
    label_to_char_map = {}

    index_offset = 0
    for idx, c in enumerate(string.digits):
      label = index_offset + idx
      char_to_label_map[c] = label
      label_to_char_map[label] = c

    index_offset += len(string.digits)
    for idx, c in enumerate(string.ascii_lowercase):
      label = index_offset + idx
      char_to_label_map[c] = label
      label_to_char_map[label] = c

    if self._case_sensitive:
      index_offset += len(string.ascii_lowercase)
      for idx, c in enumerate(string.ascii_uppercase):
        label = index_offset + idx
        char_to_label_map[c] = label
        label_to_char_map[label] = c
    else:
      for idx, c in enumerate(string.ascii_uppercase):
        label = index_offset + idx
        char_to_label_map[c] = label

    if self._include_punctuations:
      index_offset += len(string.ascii_uppercase)
      for idx, c in enumerate(string.punctuation):
        char_to_label_map[c] = label
        label_to_char_map[label] = c

    return char_to_label_map, label_to_char_map

  def text_to_labels(self, text):
    """Convert text strings to label sequences.
    Args:
      text: ascii encoded string tensor of shape [batch_size]
    """
    chars = tf.string_split(strings, delimiter='')

    chars.values()

    map_fn()


  def labels_to_text(self, labels):
    """Convert labels to text strings.
    Args:
      labels: int32 tensor []
    """
    pass
