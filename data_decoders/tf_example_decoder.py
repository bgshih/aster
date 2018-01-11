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
      fields.TfExampleFields.keypoints: \
        tf.VarLenFeature(tf.float32),
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
        slim_example_decoder.Tensor(fields.TfExampleFields.transcript),
      fields.InputDataFields.groundtruth_keypoints: \
        slim_example_decoder.Tensor(fields.TfExampleFields.keypoints)
    }

  def Decode(self, tf_example_string_tensor):
    serialized_example = tf.reshape(tf_example_string_tensor, shape=[])
    decoder = slim_example_decoder.TFExampleDecoder(self.keys_to_features,
                                                    self.items_to_handlers)
    keys = decoder.list_items()
    tensors = decoder.decode(serialized_example, items=keys)
    tensor_dict = dict(zip(keys, tensors))

    # normalize groundtruth keypoints
    # TODO: either move this to dataset creation or add image height and witdh
    image = tensor_dict[fields.InputDataFields.image]
    image_size = tf.shape(image)[:2]
    keypoints = tensor_dict[fields.InputDataFields.groundtruth_keypoints]
    num_keypoints = tf.shape(keypoints)[0] // 2
    dividor = tf.tile(
      tf.to_float(tf.stack([image_size[1], image_size[0]])),
      tf.expand_dims(num_keypoints, 0)
    )
    normalized_keypoints = tf.truediv(keypoints, dividor)
    tensor_dict[fields.InputDataFields.groundtruth_keypoints] = normalized_keypoints

    return tensor_dict
