"""Input reader builder.

Creates data sources for DetectionModels from an InputReader config. See
input_reader.proto for options.

Note: If users wishes to also use their own InputReaders with the Object
Detection configuration framework, they should define their own builder function
that wraps the build function.
"""
import os

import tensorflow as tf

from rare.core import standard_fields as fields
from rare.core import input_reader_pb2

parallel_reader = tf.contrib.slim.parallel_reader
slim_example_decoder = tf.contrib.slim.tfexample_decoder


class TfExampleDecoder(object):
  """Tensorflow Example proto decoder."""

  def __init__(self):
    self.keys_to_features = {
      fields.TfExampleFields.image_encoded: \
        tf.FixedLenFeature((), tf.string, default_value=''),
      fields.TfExampleFields.image_format: \
        tf.FixedLenFeature((), tf.string, default_value='jpeg'),
      fields.TfExampleFields.filename: \
        tf.FixedLenFeature((), tf.string, default_value=''),
      fields.TfExampleFields.source_id: \
        tf.FixedLenFeature((), tf.string, default_value=''),
      fields.TfExampleFields.height: \
        tf.FixedLenFeature((), tf.int64, default_value=1),
      fields.TfExampleFields.width: \
        tf.FixedLenFeature((), tf.int64, default_value=1),
      fields.TfExampleFields.transcript: \
        tf.FixedLenFeature((), tf.string, default_value=''),
    }
    self.items_to_handlers = {
      fields.InputDataFields.image: \
        slim_example_decoder.Image(
          image_key=fields.TfExampleFields.image_encoded,
          format_key=fields.TfExampleFields.image_format,
          channels=3
        ),
      fields.InputDataFields.filename: \
        slim_example_decoder.Tensor(fields.TfExampleFields.filename),
      fields.InputDataFields.groundtruth_text: \
        slim_example_decoder.Tensor(fields.TfExampleFields.transcript)
    }

  def Decode(self, tf_example_string_tensor):
    serialized_example = tf.reshape(tf_example_string_tensor, shape=[])
    decoder = slim_example_decoder.TFExampleDecoder(self.keys_to_features,
                                                    self.items_to_handlers)
    keys = decoder.list_items()
    tensors = decoder.decode(serialized_example, items=keys)
    tensor_dict = dict(zip(keys, tensors))
    return tensor_dict


def build(input_reader_config):
  """Builds a tensor dictionary based on the InputReader config.

  Args:
    input_reader_config: A input_reader_pb2.InputReader object.

  Returns:
    A tensor dict based on the input_reader_config.

  Raises:
    ValueError: On invalid input reader proto.
  """
  if not isinstance(input_reader_config, input_reader_pb2.InputReader):
    raise ValueError('input_reader_config not of type '
                     'input_reader_pb2.InputReader.')

  input_reader_oneof = input_reader_config.WhichOneof('input_reader')
  if input_reader_oneof == 'tf_record_input_reader':
    config = input_reader_config.tf_record_input_reader
    if not os.path.exists(config.input_path):
      raise ValueError('Input path not found: {}'.format(config.input_path))

    _, string_tensor = parallel_reader.parallel_read(
        config.input_path,
        reader_class=tf.TFRecordReader,
        num_epochs=(input_reader_config.num_epochs
                    if input_reader_config.num_epochs else None),
        num_readers=input_reader_config.num_readers,
        shuffle=input_reader_config.shuffle,
        dtypes=[tf.string, tf.string],
        capacity=input_reader_config.queue_capacity,
        min_after_dequeue=input_reader_config.min_after_dequeue)

    return TfExampleDecoder().Decode(string_tensor)

  raise ValueError('Unsupported input_reader_config: {}'.format(input_reader_oneof))
