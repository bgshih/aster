import tensorflow as tf

from rare.core import preprocessor
from rare.protos import preprocessor_pb2
from rare.builders import label_map_builder


def _get_step_config_from_proto(preprocessor_step_config, step_name):
  """Returns the value of a field named step_name from proto.

  Args:
    preprocessor_step_config: A preprocessor_pb2.PreprocessingStep object.
    step_name: Name of the field to get value from.

  Returns:
    result_dict: a sub proto message from preprocessor_step_config which will be
                 later converted to a dictionary.

  Raises:
    ValueError: If field does not exist in proto.
  """
  for field, value in preprocessor_step_config.ListFields():
    if field.name == step_name:
      return value


def _get_dict_from_proto(config):
  """Helper function to put all proto fields into a dictionary.

  For many preprocessing steps, there's an trivial 1-1 mapping from proto fields
  to function arguments. This function automatically populates a dictionary with
  the arguments from the proto.

  Protos that CANNOT be trivially populated include:
  * nested messages.
  * steps that check if an optional field is set (ie. where None != 0).
  * protos that don't map 1-1 to arguments (ie. list should be reshaped).
  * fields requiring additional validation (ie. repeated field has n elements).

  Args:
    config: A protobuf object that does not violate the conditions above.

  Returns:
    result_dict: |config| converted into a python dictionary.
  """
  result_dict = {}
  for field, value in config.ListFields():
    result_dict[field.name] = value
  return result_dict


PREPROCESSING_FUNCTION_MAP = {
    'normalize_image': preprocessor.normalize_image,
    'random_pixel_value_scale': preprocessor.random_pixel_value_scale,
    'random_rgb_to_gray': preprocessor.random_rgb_to_gray,
    'random_adjust_brightness': preprocessor.random_adjust_brightness,
    'random_adjust_contrast': preprocessor.random_adjust_contrast,
    'random_adjust_hue': preprocessor.random_adjust_hue,
    'random_adjust_saturation': preprocessor.random_adjust_saturation,
    'random_distort_color': preprocessor.random_distort_color,
    'image_to_float': preprocessor.image_to_float,
    'subtract_channel_mean': preprocessor.subtract_channel_mean,
    'rgb_to_gray': preprocessor.rgb_to_gray,
    # 'string_filtering': preprocessor.string_filtering,
}


# A map to convert from preprocessor_pb2.ResizeImage.Method enum to
# tf.image.ResizeMethod.
RESIZE_METHOD_MAP = {
    preprocessor_pb2.ResizeImage.AREA: tf.image.ResizeMethod.AREA,
    preprocessor_pb2.ResizeImage.BICUBIC: tf.image.ResizeMethod.BICUBIC,
    preprocessor_pb2.ResizeImage.BILINEAR: tf.image.ResizeMethod.BILINEAR,
    preprocessor_pb2.ResizeImage.NEAREST_NEIGHBOR: (
        tf.image.ResizeMethod.NEAREST_NEIGHBOR),
}


def build(preprocessor_step_config):
  """Builds preprocessing step based on the configuration.

  Args:
    preprocessor_step_config: PreprocessingStep configuration proto.

  Returns:
    function, argmap: A callable function and an argument map to call function
                      with.

  Raises:
    ValueError: On invalid configuration.
  """
  step_type = preprocessor_step_config.WhichOneof('preprocessing_step')

  if step_type in PREPROCESSING_FUNCTION_MAP:
    preprocessing_function = PREPROCESSING_FUNCTION_MAP[step_type]
    step_config = _get_step_config_from_proto(preprocessor_step_config,
                                              step_type)
    function_args = _get_dict_from_proto(step_config)
    return (preprocessing_function, function_args)

  if step_type == 'resize_image_random_method':
    config = preprocessor_step_config.resize_image_random_method
    return (preprocessor.resize_image_random_method,
            {
                'target_size': [config.target_height, config.target_width]
            })
  
  if step_type == 'resize_image':
    config = preprocessor_step_config.resize_image
    method = RESIZE_METHOD_MAP[config.method]
    return (preprocessor.resize_image,
            {
                'target_size': [config.target_height, config.target_width],
                'method': method
            })

  if step_type == 'string_filtering':
    config = preprocessor_step_config.string_filtering
    include_charset_list = label_map_builder._build_character_set(config.include_charset)
    include_charset = ''.join(include_charset_list)
    return (preprocessor.string_filtering,
            {
              'lower_case': config.lower_case,
              'include_charset': include_charset
            })

  raise ValueError('Unknown preprocessing step: {}'.format(step_type))
