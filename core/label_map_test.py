import tensorflow as tf

from google.protobuf import text_format
from rare.core import label_map, label_map_pb2


class LabelMapTest(tf.test.TestCase):

  def test_build_label_map(self):
    label_map_text_proto = """
    character_set {
      text_string: "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ"
      delimiter: ""
    }
    label_offset: 3
    unk_label: -2
    """
    label_map_proto = label_map_pb2.LabelMap()
    text_format.Merge(label_map_text_proto, label_map_proto)
    label_map_object = label_map.build(label_map_proto)

    test_text = tf.constant(
      ['a', 'b', '', 'abz', '0a='],
      tf.string
    )
    test_labels, text_lengths = label_map_object.text_to_labels(test_text, return_lengths=True)
    test_text_from_labels = label_map_object.labels_to_text(test_labels)

    with self.test_session() as sess:
      tf.tables_initializer().run()
      outputs = sess.run({
        'test_labels': test_labels,
        'text_lengths': text_lengths,
        'text_from_labels': test_text_from_labels
      })
      self.assertAllEqual(
        outputs['test_labels'],
        [[3, -1, -1],
         [4, -1, -1],
         [-1, -1, -1],
         [3, 4, 28],
         [-2, 3, -2]]
      )
      self.assertAllEqual(
        outputs['text_lengths'],
        [1, 1, 0, 3, 3]
      )
      self.assertAllEqual(
        outputs['text_from_labels'],
        [b'a', b'b', b'', b'abz', b'a']
      )


if __name__ == '__main__':
  tf.test.main()
