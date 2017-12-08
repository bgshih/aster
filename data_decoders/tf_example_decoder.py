import tensorflow as tf

from rare.core import standard_fields as fields

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
