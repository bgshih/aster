import tensorflow as tf
import numpy as np

from rare.c_ops import ops

class OpsTest(tf.test.TestCase):

  def test_string_reverse(self):
    test_input_strings = tf.constant(
      [b'Hello', b'world', b'1l08ck`-?=1', b''])
    test_reversed_strings = ops.string_reverse(test_input_strings)

    with self.test_session() as sess:
      self.assertAllEqual(
        test_reversed_strings.eval(),
        np.asarray([b'olleH', b'dlrow', b'1=?-`kc80l1', b''])
      )
  
  def test_divide_curve(self):
    num_keypoints = 128
    fit_points = np.array([
      [0.0, 1.0],
      [1.0, 2.0],
      [2.0, 1.0]
    ], dtype=np.float32)
    coeffs = np.polyfit(fit_points[:,0], fit_points[:,1], 2)
    poly_fn = np.poly1d(coeffs)
    xmin, xmax = np.min(fit_points[:,0]), np.max(fit_points[:,0])
    xs = np.linspace(xmin, xmax, num=(num_keypoints // 2))
    ys = poly_fn(xs)
    curve_points = np.stack([xs, ys], axis=1).flatten()
    curve_points = np.expand_dims(curve_points, axis=0)

    key_points = ops.divide_curve(curve_points, num_key_points=20)
    with self.test_session() as sess:
      sess_outputs = sess.run({
        'key_points': key_points
      })
      self.assertAllEqual(sess_outputs['key_points'].shape, (1, 40))

if __name__ == '__main__':
  tf.test.main()
