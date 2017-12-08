import string

import tensorflow as tf

from rare.core import label_map


class LabelMapTest(tf.test.TestCase):

  def test_map_text_to_labels(self):
    with tf.device('/cpu:0'):
      character_set = list(string.ascii_letters)
      test_label_map = label_map.LabelMap(
        character_set=character_set)
      test_text = tf.constant(
        ['a', 'b', '', 'abz', '0a='],
        tf.string
      )
      test_labels = test_label_map.text_to_labels(test_text)
      test_text_from_labels = test_label_map.labels_to_text(test_labels)

    with self.test_session() as sess:
      tf.tables_initializer().run()
      self.assertAllEqual(
        test_labels.eval(),
        [[3, 0, 0, 0],
         [4, 0, 0, 0],
         [0, 0, 0, 0],
         [3, 4, 28, 0],
         [2, 3, 2, 0]]
      )
      self.assertAllEqual(
        test_text_from_labels.eval(),
        [b'a', b'b', b'', b'abz', b'a']
      )


if __name__ == '__main__':
  tf.test.main()
