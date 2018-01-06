import tensorflow as tf
import numpy as np

from rare.utils import shape_utils


eps = 1e-6

class SpatialTransformer(object):

  def __init__(self,
               output_size=None,
               num_control_points=None,
               margin=0.05):
    self._output_size = output_size
    self._num_control_points = num_control_points
    self._output_grid = self._build_output_grid()
    self._output_ctrl_pts = self._build_output_control_points(margin)
    self._inv_delta_c = self._build_helper_constants()

  def preprocess(self, images):
    pass

  def transform(self, images):
    with tf.variable_scope('LocalizationNet', [images]):
      preprocessed_images = self.preprocess(images)
      input_control_points = self._localize(preprocessed_images)
    with tf.name_scope('GridGenerator', [input_control_points]):
      sampling_grid = self._generate_grid(input_control_points)
    with tf.name_scope('Sampler', [sampling_grid, images]):
      rectified_images = self._sample(sampling_grid, images)
    return rectified_images

  def _localize(self, preprocessed_images):
    pass

  def _generate_grid(self, input_ctrl_pts):
    # compute transformation
    C = tf.constant(self._output_ctrl_pts, tf.float32)
    Cp = input_ctrl_pts
    T = tf.matmul(tf.constant(self._inv_delta_c, dtype=tf.float32),
                  tf.concat([Cp, tf.zeros([3, 2])], axis=0))

    # transform
    k = self._num_control_points
    output_grid_points = self._output_grid.reshape([-1, 2])
    n = output_grid_points.shape[0]
    P = tf.constant(output_grid_points, tf.float32)
    P_tile = tf.tile(tf.expand_dims(P, axis=1), [1, k, 1]) # => [n, k, 2]
    C_tile = tf.expand_dims(C, axis=0) # => [1, k, 2]
    P_diff = P_tile - C_tile   # => [n, k, 2]
    P_norm = tf.norm(P_diff, axis=2, ord=2, keep_dims=False) # => [n, k]

    rbf = tf.multiply(tf.square(P_norm), tf.log(P_norm + eps)) # => [n, k]
    P_lifted = tf.concat([ tf.ones([n, 1]), P, rbf ], axis=1)
    Pp = tf.matmul(P_lifted, T)
    return Pp

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
  
  def _sample(self, image, sampling_grid):
    sampling_grid = tf.maximum(0.0, tf.minimum(1.0-eps, sampling_grid))
    orig_dytpe = image.dtype
    image = tf.to_float(image)
    image_h, image_w, _ = shape_utils.combined_static_and_dynamic_shape(image)
    Gx = image_w * sampling_grid[:,0]
    Gy = image_h * sampling_grid[:,1]
    Gx0 = tf.cast(tf.floor(Gx), tf.int32)
    Gx1 = Gx0 + 1
    Gy0 = tf.cast(tf.floor(Gy), tf.int32)
    Gy1 = Gy0 + 1

    I00 = tf.gather_nd(image, tf.stack([Gy0, Gx0], axis=1))
    I01 = tf.gather_nd(image, tf.stack([Gy1, Gx0], axis=1))
    I10 = tf.gather_nd(image, tf.stack([Gy0, Gx1], axis=1))
    I11 = tf.gather_nd(image, tf.stack([Gy1, Gx1], axis=1))

    Gx0 = tf.to_float(Gx0)
    Gx1 = tf.to_float(Gx1)
    Gy0 = tf.to_float(Gy0)
    Gy1 = tf.to_float(Gy1)

    w00 = (Gx1 - Gx) * (Gy1 - Gy)
    w01 = (Gx1 - Gx) * (Gy - Gy0)
    w10 = (Gx - Gx0) * (Gy1 - Gy)
    w11 = (Gx - Gx0) * (Gy - Gy0)

    pixels = tf.add_n([
      tf.expand_dims(w00, axis=1) * I00,
      tf.expand_dims(w01, axis=1) * I01,
      tf.expand_dims(w10, axis=1) * I10,
      tf.expand_dims(w11, axis=1) * I11,
    ])
    output_h, output_w = self._output_size
    output_map = tf.reshape(pixels, [output_h, output_w, -1])
    output_map = tf.cast(output_map, dtype=orig_dytpe)

    return output_map

  def _batch_sample(self, images, batch_sampling_grid):
    """
    Args:
      images: tensor of any time with shape [batch_size, image_h, image_w, depth]
      batch_sampling_grid; float32 tensor with shape [batch_size, num_sampling_pts, 2]
    """
    batch_G = tf.maximum(0.0, tf.minimum(1.0-eps, batch_sampling_grid)) # => [B, n, 2]
    orig_dytpe = images.dtype
    images = tf.to_float(images)
    batch_size, image_h, image_w, _ = shape_utils.combined_static_and_dynamic_shape(images)
    n = shape_utils.combined_static_and_dynamic_shape(batch_sampling_grid)[1]

    batch_Gx = image_w * batch_G[:,:,0]
    batch_Gy = image_h * batch_G[:,:,1]
    batch_Gx0 = tf.cast(tf.floor(batch_Gx), tf.int32)
    batch_Gx1 = batch_Gx0 + 1
    batch_Gy0 = tf.cast(tf.floor(batch_Gy), tf.int32)
    batch_Gy1 = batch_Gy0 + 1
    # G* => [batch_size, n, 2]
    # G*x, G*y => [batch_size, n]

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

    output_h, output_w = self._output_size
    output_maps = tf.reshape(batch_pixels, [batch_size, output_h, output_w, -1])
    output_maps = tf.cast(output_maps, dtype=orig_dytpe)
    return output_maps

  def _build_output_grid(self):
    output_h, output_w = self._output_size
    output_grid_x = (np.arange(output_w) + 0.5) / output_w
    output_grid_y = (np.arange(output_h) + 0.5) / output_h
    output_grid = np.stack(
      np.meshgrid(output_grid_x, output_grid_y),
      axis=2)
    return output_grid

  def _build_output_control_points(self, margin):
    num_ctrl_pts_per_side = self._num_control_points // 2
    ctrl_pts_x = np.linspace(margin, 1-margin, num_ctrl_pts_per_side)
    ctrl_pts_y_top = np.ones(num_ctrl_pts_per_side) * margin
    ctrl_pts_y_bottom = np.ones(num_ctrl_pts_per_side) * (1 - margin)
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
