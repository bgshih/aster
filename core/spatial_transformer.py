import tensorflow as tf
import numpy as np
from tensorflow.contrib.layers import avg_pool2d, fully_connected
from tensorflow.contrib.framework import arg_scope

from rare.utils import shape_utils


eps = 1e-6

class SpatialTransformer(object):

  def __init__(self,
               convnet=None,
               fc_hyperparams=None,
               localization_image_size=None,
               output_image_size=None,
               num_control_points=None,
               init_bias_pattern=None,
               summarize_activations=False):
    self._convnet = convnet
    self._fc_hyperparams = fc_hyperparams
    self._localization_image_size = localization_image_size
    self._output_image_size = output_image_size
    self._num_control_points = num_control_points
    self._init_bias_pattern = init_bias_pattern
    self._summarize_activations = summarize_activations

    self._output_grid = self._build_output_grid()
    self._output_ctrl_pts = self._build_output_control_points()
    self._inv_delta_c = self._build_helper_constants()

    if self._init_bias_pattern == 'slope':
      self._init_bias = self._build_init_bias_slope_pattern()
    elif self._init_bias_pattern == 'identity':
      self._init_bias = self._build_init_bias_identity_pattern()
    elif self._init_bias_pattern == 'sine':
      self._init_bias = self._build_init_bias_sine_pattern()
    elif self._init_bias_pattern == 'random':
      self._init_bias = None
    else:
      raise ValueError('Unknown init bias pattern: {}'.format(self._init_bias_pattern))

  def batch_transform(self, preprocessed_inputs):
    with tf.variable_scope('LocalizationNet', [preprocessed_inputs]):
      resized_images = tf.image.resize_images(preprocessed_inputs, self._localization_image_size)
      # preprocessed_images = self._preprocess(resized_images)
      input_control_points = self._localize(resized_images)
    
    with tf.name_scope('GridGenerator', [input_control_points]):
      sampling_grid = self._batch_generate_grid(input_control_points)

    with tf.name_scope('Sampler', [sampling_grid, preprocessed_inputs]):
      rectified_images = self._batch_sample(preprocessed_inputs, sampling_grid)

    return {
      'rectified_images': rectified_images,
      'control_points': input_control_points
    }

  def _preprocess(self, resized_inputs):
    return self._convnet.preprocess(resized_inputs)

  def _localize(self, preprocessed_images):
    k = self._num_control_points
    conv_output = self._convnet.extract_features(preprocessed_images)[-1]
    batch_size = shape_utils.combined_static_and_dynamic_shape(conv_output)[0]
    conv_output = tf.reshape(conv_output, [batch_size, -1])
    with arg_scope(self._fc_hyperparams):
      fc1 = fully_connected(conv_output, 512)
      # fc2 = fully_connected(fc1, 2 * k, activation_fn=None, normalizer_fn=None)
      # ctrl_pts = tf.sigmoid(fc2)
      fc2_weights_initializer = tf.zeros_initializer()
      fc2_biases_initializer = tf.constant_initializer(self._init_bias)
      fc2 = fully_connected(0.1 * fc1, 2 * k,
        weights_initializer=fc2_weights_initializer,
        biases_initializer=fc2_biases_initializer,
        activation_fn=None,
        normalizer_fn=None)
      if self._summarize_activations:
        tf.summary.histogram('fc1', fc1)
        tf.summary.histogram('fc2', fc2)
    # ctrl_pts = tf.sigmoid(fc2)
    ctrl_pts = fc2
    ctrl_pts = tf.reshape(ctrl_pts, [batch_size, k, 2])
    return ctrl_pts

  def _batch_generate_grid(self, input_ctrl_pts):
    """
    Args
      input_ctrl_pts: float32 tensor of shape [batch_size, num_ctrl_pts, 2]
    Returns
      sampling_grid: float32 tensor of shape [num_sampling_pts, 2]
    """
    C = tf.constant(self._output_ctrl_pts, tf.float32) # => [k, 2]
    batch_Cp = input_ctrl_pts # => [B, k, 2]
    batch_size = input_ctrl_pts.shape[0]

    inv_delta_c = tf.constant(self._inv_delta_c, dtype=tf.float32)
    batch_inv_delta_c = tf.tile(
      tf.expand_dims(inv_delta_c, 0),
      [batch_size, 1, 1]) # => [B, k+3, k+3]
    batch_Cp_zero = tf.concat(
      [batch_Cp, tf.zeros([batch_size, 3, 2])],
      axis=1) # => [B, k+3, 2]
    batch_T = tf.matmul(batch_inv_delta_c, batch_Cp_zero) # => [B, k+3, 2]

    k = self._num_control_points
    G = tf.constant(self._output_grid.reshape([-1, 2]), tf.float32) # => [n, 2]
    n = G.shape[0]
    
    G_tile = tf.tile(tf.expand_dims(G, axis=1), [1,k,1]) # => [n,k,2]
    C_tile = tf.expand_dims(C, axis=0) # => [1, k, 2]
    G_diff = G_tile - C_tile   # => [n, k, 2]
    rbf_norm = tf.norm(G_diff, axis=2, ord=2, keep_dims=False) # => [n, k]
    rbf = tf.multiply(tf.square(rbf_norm), tf.log(rbf_norm + eps)) # => [n, k]
    G_lifted = tf.concat([ tf.ones([n, 1]), G, rbf ], axis=1) # => [n, k+3]
    batch_G_lifted = tf.tile(tf.expand_dims(G_lifted, 0), [batch_size,1,1]) # => [B, n, k+3]

    batch_Gp = tf.matmul(batch_G_lifted, batch_T)
    return batch_Gp
  
  def _batch_sample(self, images, batch_sampling_grid):
    """
    Args:
      images: tensor of any time with shape [batch_size, image_h, image_w, depth]
      batch_sampling_grid; float32 tensor with shape [batch_size, num_sampling_pts, 2]
    """
    if images.dtype != tf.float32:
      raise ValueError('image must be of type tf.float32')
    batch_G = batch_sampling_grid
    batch_size, image_h, image_w, _ = shape_utils.combined_static_and_dynamic_shape(images)
    n = shape_utils.combined_static_and_dynamic_shape(batch_sampling_grid)[1]

    batch_Gx = image_w * batch_G[:,:,0]
    batch_Gy = image_h * batch_G[:,:,1]
    batch_Gx = tf.clip_by_value(batch_Gx, 0., image_w-2)
    batch_Gy = tf.clip_by_value(batch_Gy, 0., image_h-2)

    batch_Gx0 = tf.cast(tf.floor(batch_Gx), tf.int32) # G* => [batch_size, n, 2]
    batch_Gx1 = batch_Gx0 + 1 # G*x, G*y => [batch_size, n]
    batch_Gy0 = tf.cast(tf.floor(batch_Gy), tf.int32)
    batch_Gy1 = batch_Gy0 + 1

    def _get_pixels(images, batch_x, batch_y, batch_indices):
      indices = tf.stack([batch_indices, batch_y, batch_x], axis=2) # => [B, n, 3]
      pixels = tf.gather_nd(images, indices)
      return pixels

    batch_indices = tf.tile(
      tf.expand_dims(tf.range(batch_size), 1),
      [1, n]) # => [B, n]
    batch_I00 = _get_pixels(images, batch_Gx0, batch_Gy0, batch_indices)
    batch_I01 = _get_pixels(images, batch_Gx0, batch_Gy1, batch_indices)
    batch_I10 = _get_pixels(images, batch_Gx1, batch_Gy0, batch_indices)
    batch_I11 = _get_pixels(images, batch_Gx1, batch_Gy1, batch_indices) # => [B, n, d]

    batch_Gx0 = tf.to_float(batch_Gx0)
    batch_Gx1 = tf.to_float(batch_Gx1)
    batch_Gy0 = tf.to_float(batch_Gy0)
    batch_Gy1 = tf.to_float(batch_Gy1)

    batch_w00 = (batch_Gx1 - batch_Gx) * (batch_Gy1 - batch_Gy)
    batch_w01 = (batch_Gx1 - batch_Gx) * (batch_Gy - batch_Gy0)
    batch_w10 = (batch_Gx - batch_Gx0) * (batch_Gy1 - batch_Gy)
    batch_w11 = (batch_Gx - batch_Gx0) * (batch_Gy - batch_Gy0) # => [B, n]

    batch_pixels = tf.add_n([
      tf.expand_dims(batch_w00, axis=2) * batch_I00,
      tf.expand_dims(batch_w01, axis=2) * batch_I01,
      tf.expand_dims(batch_w10, axis=2) * batch_I10,
      tf.expand_dims(batch_w11, axis=2) * batch_I11,
    ])

    output_h, output_w = self._output_image_size
    output_maps = tf.reshape(batch_pixels, [batch_size, output_h, output_w, -1])
    output_maps = tf.cast(output_maps, dtype=images.dtype)

    if self._summarize_activations:
      tf.summary.image('InputImage1', images[:2], max_outputs=2)
      tf.summary.image('InputImage2', images[-2:], max_outputs=2)
      tf.summary.image('RectifiedImage1', output_maps[:2], max_outputs=2)
      tf.summary.image('RectifiedImage2', output_maps[-2:], max_outputs=2)

    return output_maps

  def _build_output_grid(self):
    output_h, output_w = self._output_image_size
    output_grid_x = (np.arange(output_w) + 0.5) / output_w
    output_grid_y = (np.arange(output_h) + 0.5) / output_h
    output_grid = np.stack(
      np.meshgrid(output_grid_x, output_grid_y),
      axis=2)
    return output_grid

  def _build_output_control_points(self):
    num_ctrl_pts_per_side = self._num_control_points // 2
    ctrl_pts_x = np.linspace(0., 1., num_ctrl_pts_per_side)
    ctrl_pts_y_top = np.ones(num_ctrl_pts_per_side) * 0.
    ctrl_pts_y_bottom = np.ones(num_ctrl_pts_per_side) * 1.
    ctrl_pts_top = np.stack([ctrl_pts_x, ctrl_pts_y_top], axis=1)
    ctrl_pts_bottom = np.stack([ctrl_pts_x, ctrl_pts_y_bottom], axis=1)
    output_ctrl_pts = np.concatenate([ctrl_pts_top, ctrl_pts_bottom], axis=0)
    return output_ctrl_pts

  def _build_helper_constants(self):
    C = self._output_ctrl_pts
    k = self._num_control_points
    hat_C = np.zeros((k, k), dtype=float)
    for i in range(k):
      for j in range(k):
        hat_C[i,j] = np.linalg.norm(C[i] - C[j])
    np.fill_diagonal(hat_C, 1)
    hat_C = (hat_C ** 2) * np.log(hat_C)
    delta_C = np.concatenate(
      [
        np.concatenate([ np.ones((k, 1)), C, hat_C ], axis=1),
        np.concatenate([ np.zeros((2, 3)), np.transpose(C) ], axis=1),
        np.concatenate([ np.zeros((1, 3)), np.ones((1, k)) ], axis=1)
      ],
      axis=0
    )
    inv_delta_C = np.linalg.inv(delta_C)
    return inv_delta_C

  def _build_init_bias_slope_pattern(self):
    num_ctrl_pts_per_side = self._num_control_points // 2
    upper_x = np.linspace(0., 1, num=num_ctrl_pts_per_side)
    upper_y = np.linspace(0., 0.3, num=num_ctrl_pts_per_side)
    lower_x = np.linspace(0., 1., num=num_ctrl_pts_per_side)
    lower_y = np.linspace(0.7, 1., num=num_ctrl_pts_per_side)
    init_ctrl_pts = np.concatenate([
      np.stack([upper_x, upper_y], axis=1),
      np.stack([lower_x, lower_y], axis=1),
    ], axis=0)
    init_biases = -np.log(1. / init_ctrl_pts - 1.)
    return init_biases

  def _build_init_bias_identity_pattern(self):
    num_ctrl_pts_per_side = self._num_control_points // 2
    upper_x = np.linspace(0.0, 1.0, num=num_ctrl_pts_per_side)
    upper_y = np.linspace(0.0, 0.0, num=num_ctrl_pts_per_side)
    lower_x = np.linspace(0.0, 1.0, num=num_ctrl_pts_per_side)
    lower_y = np.linspace(1.0, 1.0, num=num_ctrl_pts_per_side)
    init_ctrl_pts = np.concatenate([
      np.stack([upper_x, upper_y], axis=1),
      np.stack([lower_x, lower_y], axis=1),
    ], axis=0)
    init_biases = -np.log(1. / init_ctrl_pts - 1.)
    return init_biases

  def _build_init_bias_sine_pattern(self, logit=False):
    num_ctrl_pts_per_side = self._num_control_points // 2
    upper_x = np.linspace(0.05, 0.95, num=num_ctrl_pts_per_side)
    upper_y = 0.25 + 0.2 * np.sin(2 * np.pi * upper_x)
    lower_x = np.linspace(0.05, 0.95, num=num_ctrl_pts_per_side)
    lower_y = 0.75 + 0.2 * np.sin(2 * np.pi * lower_x)
    init_ctrl_pts = np.concatenate([
      np.stack([upper_x, upper_y], axis=1),
      np.stack([lower_x, lower_y], axis=1),
    ], axis=0)
    if logit:
      init_biases = -np.log(1. / init_ctrl_pts - 1.)
    else:
      init_biases = init_ctrl_pts
    return init_biases
