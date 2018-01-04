import tensorflow as tf
import numpy as np


class SpatialTransformer(object):

  def __init__(self,
               output_size=None,
               num_control_points=None,
               margin=0.05):
    self._output_size = output_size
    self._num_control_points = num_control_points
    self._output_grid = self._build_output_grid()
    self._output_ctrl_pts = self._build_output_control_points()

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

  def _generate_grid(self, cp):
    """
    Args:
      cp: [num_pts, 2]
    """
    c = self._output_ctrl_pts
    k = c.get_shape()[0].value
    cp1 = tf.tile(tf.expand_dims(cp, axis=1), [1, k, 1])
    c2 = tf.tile(tf.expand_dims(c, axis=0), [k, 1, 1])
    dist = tf.norm(cp1 - c2, axis=1)
    rbf = tf.square(dist) * tf.log(dist)
    c_lifted = tf.concat(
      [
        tf.ones([1, k]),
        tf.transpose(c),
        rbf
      ],
      axis=0
    )
    T = tf.concat([])
  
  def _sample(self, image, input_grid, ):
    pass

  def _build_output_grid(self):
    output_h, output_w = self._output_size
    output_grid_x = np.range(output_w) + 0.5
    output_grid_y = np.range(output_h) + 0.5
    output_grid = np.stack(
      np.mesh_grid(output_grid_x, output_grid_y),
      axis=2)
    return tf.constant(output_grid, tf.float32)

  def _build_output_control_points(self):
    num_ctrl_pts_per_side = self._num_control_points / 2
    ctrl_pts_x = np.linspace(margin, 1-margin, num_ctrl_pts_per_side)
    ctrl_pts_y_top = np.empty(num_ctrl_pts_per_side).fill(margin)
    ctrl_pts_y_bottom = np.empty(num_ctrl_pts_per_side).fill(1-margin)
    ctrl_pts_top = np.stack([ctrl_pts_x, ctrl_pts_y_top], axis=1)
    ctrl_pts_bottom = np.stack([ctrl_pts_x, ctrl_pts_y_bottom], axis=1)
    output_ctrl_pts = np.concatenate([ctrl_pts_top, ctrl_pts_bottom], axis=0)
    return tf.constant(output_ctrl_pts, tf.float32)

  def _build_helper_constants(self, c):
    # c: [num_pts, 2]
    k = c.shape[0]
    hat_c = np.zeros((k, k), dtype=float)
    for i in range(k):
      for j in range(k):
        hat_c[i,j] = np.linalg.norm(c[i] - c[j])
    hat_c = (hat_c ** 2) * np.log(hat_c)
    delta_c = np.concatenate(
      [
        np.concatenate([ np.ones((1,k)), np.zeros(1, k+2) ], axis=1),
        np.concatenate([ c, np.zeros(k, 3) ], axis=1),
        np.concatenate([ hat_c, np.ones((k, 1)), np.transpose(c) ], axis=1),
      ],
      axis=0
    )
    inv_delta_c = np.linalg.inv(delta_c)
    