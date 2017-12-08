import tensorflow as tf

from rare.core import label_map


class LabelMapTest(tf.test.TestCase):

  def test_map_text_to_labels(self):
    test_label_map = label_map.LabelMap(dictionary)

    test_text = ["阿斯顿发", "asdf"]
    test_text_tensor = tf.constant(
      [t.encode('utf-8') for t in test_text],
      tf.string)
    with self.test_session() as sess:
      print(test_text_tensor.eval())


if __name__ == '__main__':
  tf.test.main()
