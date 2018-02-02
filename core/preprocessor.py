import sys
import string

import tensorflow as tf
from tensorflow.python.ops import control_flow_ops

from rare.core import standard_fields as fields
from rare.c_ops import ops


def _apply_with_random_selector(x, func, num_cases):
  """Computes func(x, sel), with sel sampled from [0...num_cases-1].

  Args:
    x: input Tensor.
    func: Python function to apply.
    num_cases: Python int32, number of cases to sample sel from.

  Returns:
    The result of func(x, sel), where func receives the value of the
    selector as a python integer, but sel is sampled dynamically.
  """
  rand_sel = tf.random_uniform([], maxval=num_cases, dtype=tf.int32)
  # Pass the real x only to one of the func calls.
  return control_flow_ops.merge([func(
      control_flow_ops.switch(x, tf.equal(rand_sel, case))[1], case)
                                 for case in range(num_cases)])[0]


def _apply_with_random_selector_tuples(x, func, num_cases):
  """Computes func(x, sel), with sel sampled from [0...num_cases-1].

  Args:
    x: A tuple of input tensors.
    func: Python function to apply.
    num_cases: Python int32, number of cases to sample sel from.

  Returns:
    The result of func(x, sel), where func receives the value of the
    selector as a python integer, but sel is sampled dynamically.
  """
  num_inputs = len(x)
  rand_sel = tf.random_uniform([], maxval=num_cases, dtype=tf.int32)
  # Pass the real x only to one of the func calls.

  tuples = [list() for t in x]
  for case in range(num_cases):
    new_x = [control_flow_ops.switch(t, tf.equal(rand_sel, case))[1] for t in x]
    output = func(tuple(new_x), case)
    for j in range(num_inputs):
      tuples[j].append(output[j])

  for i in range(num_inputs):
    tuples[i] = control_flow_ops.merge(tuples[i])[0]
  return tuple(tuples)


def _random_integer(minval, maxval, seed):
  """Returns a random 0-D tensor between minval and maxval.

  Args:
    minval: minimum value of the random tensor.
    maxval: maximum value of the random tensor.
    seed: random seed.

  Returns:
    A random 0-D tensor between minval and maxval.
  """
  return tf.random_uniform(
      [], minval=minval, maxval=maxval, dtype=tf.int32, seed=seed)


def resize_image_random_method(image, target_size):
  """Resize image with random image interpolation method.

  Args:
    image: rank 3 tensor of shape [image_height, image_width, 3]
    target_size: [target_height, target_width]
  
  Returns:
    resized_image
  """
  with tf.name_scope('ResizeRandomMethod', values=[image]):
    # resize image
    resized_image = _apply_with_random_selector(
        image,
        lambda x, method: tf.image.resize_images(x, target_size, method),
        num_cases=4)
  return resized_image


def resize_image(image,
                 target_size,
                 method=tf.image.ResizeMethod.BILINEAR,
                 align_corners=False):
  with tf.name_scope('ResizeImage', values=[image, target_size]):
    new_image = tf.image.resize_images(image, target_size,
                                       method=method,
                                       align_corners=align_corners)
  return new_image


def normalize_image(image, original_minval, original_maxval, target_minval,
                    target_maxval):
  """Normalizes pixel values in the image.

  Moves the pixel values from the current [original_minval, original_maxval]
  range to a the [target_minval, target_maxval] range.

  Args:
    image: rank 3 float32 tensor containing 1
           image -> [height, width, channels].
    original_minval: current image minimum value.
    original_maxval: current image maximum value.
    target_minval: target image minimum value.
    target_maxval: target image maximum value.

  Returns:
    image: image which is the same shape as input image.
  """
  with tf.name_scope('NormalizeImage', values=[image]):
    original_minval = float(original_minval)
    original_maxval = float(original_maxval)
    target_minval = float(target_minval)
    target_maxval = float(target_maxval)
    image = tf.to_float(image)
    image = tf.subtract(image, original_minval)
    image = tf.multiply(image, (target_maxval - target_minval) /
                        (original_maxval - original_minval))
    image = tf.add(image, target_minval)
    return image


def random_pixel_value_scale(image, minval=0.9, maxval=1.1, seed=None):
  """Scales each value in the pixels of the image.

     This function scales each pixel independent of the other ones.
     For each value in image tensor, draws a random number between
     minval and maxval and multiples the values with them.

  Args:
    image: rank 3 float32 tensor contains 1 image -> [height, width, channels]
           with pixel values varying between [0, 1].
    minval: lower ratio of scaling pixel values.
    maxval: upper ratio of scaling pixel values.
    seed: random seed.

  Returns:
    image: image which is the same shape as input image.
    boxes: boxes which is the same shape as input boxes.
  """
  with tf.name_scope('RandomPixelValueScale', values=[image]):
    color_coef = tf.random_uniform(
        tf.shape(image),
        minval=minval,
        maxval=maxval,
        dtype=tf.float32,
        seed=seed)
    image = tf.multiply(image, color_coef)
    image = tf.clip_by_value(image, 0.0, 1.0)

  return image


def random_rgb_to_gray(image, probability=0.1, seed=None):
  """Changes the image from RGB to Grayscale with the given probability.

  Args:
    image: rank 3 float32 tensor contains 1 image -> [height, width, channels]
           with pixel values varying between [0, 1].
    probability: the probability of returning a grayscale image.
            The probability should be a number between [0, 1].
    seed: random seed.

  Returns:
    image: image which is the same shape as input image.
  """
  def _image_to_gray(image):
    image_gray1 = tf.image.rgb_to_grayscale(image)
    image_gray3 = tf.image.grayscale_to_rgb(image_gray1)
    return image_gray3

  with tf.name_scope('RandomRGBtoGray', values=[image]):
    # random variable defining whether to do flip or not
    do_gray_random = tf.random_uniform([], seed=seed)

    image = tf.cond(
        tf.greater(do_gray_random, probability), lambda: image,
        lambda: _image_to_gray(image))

  return image


def random_adjust_brightness(image, max_delta=0.2):
  """Randomly adjusts brightness.

  Makes sure the output image is still between 0 and 1.

  Args:
    image: rank 3 float32 tensor contains 1 image -> [height, width, channels]
           with pixel values varying between [0, 1].
    max_delta: how much to change the brightness. A value between [0, 1).

  Returns:
    image: image which is the same shape as input image.
    boxes: boxes which is the same shape as input boxes.
  """
  with tf.name_scope('RandomAdjustBrightness', values=[image]):
    image = tf.image.random_brightness(image, max_delta)
    image = tf.clip_by_value(image, clip_value_min=0.0, clip_value_max=1.0)
    return image


def random_adjust_contrast(image, min_delta=0.8, max_delta=1.25):
  """Randomly adjusts contrast.

  Makes sure the output image is still between 0 and 1.

  Args:
    image: rank 3 float32 tensor contains 1 image -> [height, width, channels]
           with pixel values varying between [0, 1].
    min_delta: see max_delta.
    max_delta: how much to change the contrast. Contrast will change with a
               value between min_delta and max_delta. This value will be
               multiplied to the current contrast of the image.

  Returns:
    image: image which is the same shape as input image.
  """
  with tf.name_scope('RandomAdjustContrast', values=[image]):
    image = tf.image.random_contrast(image, min_delta, max_delta)
    image = tf.clip_by_value(image, clip_value_min=0.0, clip_value_max=1.0)
    return image


def random_adjust_hue(image, max_delta=0.02):
  """Randomly adjusts hue.

  Makes sure the output image is still between 0 and 1.

  Args:
    image: rank 3 float32 tensor contains 1 image -> [height, width, channels]
           with pixel values varying between [0, 1].
    max_delta: change hue randomly with a value between 0 and max_delta.

  Returns:
    image: image which is the same shape as input image.
  """
  with tf.name_scope('RandomAdjustHue', values=[image]):
    image = tf.image.random_hue(image, max_delta)
    image = tf.clip_by_value(image, clip_value_min=0.0, clip_value_max=1.0)
    return image


def random_adjust_saturation(image, min_delta=0.8, max_delta=1.25):
  """Randomly adjusts saturation.

  Makes sure the output image is still between 0 and 1.

  Args:
    image: rank 3 float32 tensor contains 1 image -> [height, width, channels]
           with pixel values varying between [0, 1].
    min_delta: see max_delta.
    max_delta: how much to change the saturation. Saturation will change with a
               value between min_delta and max_delta. This value will be
               multiplied to the current saturation of the image.

  Returns:
    image: image which is the same shape as input image.
  """
  with tf.name_scope('RandomAdjustSaturation', values=[image]):
    image = tf.image.random_saturation(image, min_delta, max_delta)
    image = tf.clip_by_value(image, clip_value_min=0.0, clip_value_max=1.0)
    return image


def random_distort_color(image, color_ordering=0):
  """Randomly distorts color.

  Randomly distorts color using a combination of brightness, hue, contrast
  and saturation changes. Makes sure the output image is still between 0 and 1.

  Args:
    image: rank 3 float32 tensor contains 1 image -> [height, width, channels]
           with pixel values varying between [0, 1].
    color_ordering: Python int, a type of distortion (valid values: 0, 1).

  Returns:
    image: image which is the same shape as input image.

  Raises:
    ValueError: if color_ordering is not in {0, 1}.
  """
  with tf.name_scope('RandomDistortColor', values=[image]):
    if color_ordering == 0:
      image = tf.image.random_brightness(image, max_delta=32. / 255.)
      image = tf.image.random_saturation(image, lower=0.5, upper=1.5)
      image = tf.image.random_hue(image, max_delta=0.2)
      image = tf.image.random_contrast(image, lower=0.5, upper=1.5)
    elif color_ordering == 1:
      image = tf.image.random_brightness(image, max_delta=32. / 255.)
      image = tf.image.random_contrast(image, lower=0.5, upper=1.5)
      image = tf.image.random_saturation(image, lower=0.5, upper=1.5)
      image = tf.image.random_hue(image, max_delta=0.2)
    else:
      raise ValueError('color_ordering must be in {0, 1}')

    # The random_* ops do not necessarily clamp.
    image = tf.clip_by_value(image, 0.0, 1.0)
    return image


def image_to_float(image):
  """Used in Faster R-CNN. Casts image pixel values to float.

  Args:
    image: input image which might be in tf.uint8 or sth else format

  Returns:
    image: image in tf.float32 format.
  """
  with tf.name_scope('ImageToFloat', values=[image]):
    image = tf.to_float(image)
    return image


def subtract_channel_mean(image, means=None):
  """Normalizes an image by subtracting a mean from each channel.

  Args:
    image: A 3D tensor of shape [height, width, channels]
    means: float list containing a mean for each channel
  Returns:
    normalized_images: a tensor of shape [height, width, channels]
  Raises:
    ValueError: if images is not a 4D tensor or if the number of means is not
      equal to the number of channels.
  """
  with tf.name_scope('SubtractChannelMean', values=[image, means]):
    if len(image.get_shape()) != 3:
      raise ValueError('Input must be of size [height, width, channels]')
    if len(means) != image.get_shape()[-1]:
      raise ValueError('len(means) must match the number of channels')
    return image - [[means]]


def rgb_to_gray(image, three_channels=False):
  """Converts a 3 channel RGB image to a 1 channel grayscale image.

  Args:
    image: Rank 3 float32 tensor containing 1 image -> [height, width, 3]
           with pixel values varying between [0, 1].

  Returns:
    image: A single channel grayscale image -> [image, height, 1].
  """
  gray_image = tf.image.rgb_to_grayscale(image)
  if three_channels:
    gray_image = tf.tile(gray_image, [1,1,3])
  return gray_image


def string_filtering(text, lower_case=False, include_charset=""):
  return ops.string_filtering([text],
    lower_case=lower_case, include_charset=include_charset)[0]


def get_default_func_arg_map():
  prep_func_arg_map = {
      resize_image: (fields.InputDataFields.image,),
      resize_image_random_method: (fields.InputDataFields.image,),
      random_pixel_value_scale: (fields.InputDataFields.image,),
      random_rgb_to_gray: (fields.InputDataFields.image,),
      random_adjust_brightness: (fields.InputDataFields.image,),
      random_adjust_contrast: (fields.InputDataFields.image,),
      random_adjust_hue: (fields.InputDataFields.image,),
      random_adjust_saturation: (fields.InputDataFields.image,),
      random_distort_color: (fields.InputDataFields.image,),
      image_to_float: (fields.InputDataFields.image,),
      subtract_channel_mean: (fields.InputDataFields.image,),
      rgb_to_gray: (fields.InputDataFields.image,),
      string_filtering: (fields.InputDataFields.groundtruth_text,)
  }
  return prep_func_arg_map


def preprocess(tensor_dict, preprocess_options, func_arg_map=None):
  """Preprocess images and bounding boxes.

  Various types of preprocessing (to be implemented) based on the
  preprocess_options dictionary e.g. "crop image" (affects image and possibly
  boxes), "white balance image" (affects only image), etc. If self._options
  is None, no preprocessing is done.

  Args:
    tensor_dict: dictionary that contains images, boxes, and can contain other
                 things as well.
                 images-> rank 4 float32 tensor contains
                          1 image -> [1, height, width, 3].
                          with pixel values varying between [0, 1]
                 boxes-> rank 2 float32 tensor containing
                         the bounding boxes -> [N, 4].
                         Boxes are in normalized form meaning
                         their coordinates vary between [0, 1].
                         Each row is in the form
                         of [ymin, xmin, ymax, xmax].
    preprocess_options: It is a list of tuples, where each tuple contains a
                        function and a dictionary that contains arguments and
                        their values.
    func_arg_map: mapping from preprocessing functions to arguments that they
                  expect to receive and return.

  Returns:
    tensor_dict: which contains the preprocessed images, bounding boxes, etc.

  Raises:
    ValueError: (a) If the functions passed to Preprocess
                    are not in func_arg_map.
                (b) If the arguments that a function needs
                    do not exist in tensor_dict.
                (c) If image in tensor_dict is not rank 4
  """
  if func_arg_map is None:
    func_arg_map = get_default_func_arg_map()

  # changes the images to image (rank 4 to rank 3) since the functions
  # receive rank 3 tensor for image
  if fields.InputDataFields.image in tensor_dict:
    image = tensor_dict[fields.InputDataFields.image]
    # if len(images.get_shape()) != 4:
      # raise ValueError('images in tensor_dict should be rank 4')
    # image = tf.squeeze(images, squeeze_dims=[0])
    if len(image.get_shape()) != 3:
      raise ValueError('images in tensor_dict should be rank 3')
    tensor_dict[fields.InputDataFields.image] = image

  # Preprocess inputs based on preprocess_options
  for option in preprocess_options:
    func, params = option
    if func not in func_arg_map:
      raise ValueError('The function %s does not exist in func_arg_map' %
                       (func.__name__))
    arg_names = func_arg_map[func]
    for a in arg_names:
      if a is not None and a not in tensor_dict:
        raise ValueError('The function %s requires argument %s' %
                         (func.__name__, a))

    def get_arg(key):
      return tensor_dict[key] if key is not None else None
    args = [get_arg(a) for a in arg_names]
    results = func(*args, **params)
    if not isinstance(results, (list, tuple)):
      results = (results,)
    # Removes None args since the return values will not contain those.
    arg_names = [arg_name for arg_name in arg_names if arg_name is not None]
    for res, arg_name in zip(results, arg_names):
      tensor_dict[arg_name] = res

  # # changes the image to images (rank 3 to rank 4) to be compatible to what
  # # we received in the first place
  # if fields.InputDataFields.image in tensor_dict:
  #   image = tensor_dict[fields.InputDataFields.image]
  #   images = tf.expand_dims(image, 0)
  #   tensor_dict[fields.InputDataFields.image] = images

  return tensor_dict
