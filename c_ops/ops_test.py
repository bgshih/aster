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

if __name__ == '__main__':
  tf.test.main()
