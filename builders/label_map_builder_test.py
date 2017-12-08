import tensorflow as tf

from google.protobuf import text_format
from rare.builders import label_map_builder
from rare.protos import label_map_pb2


class LabelMapBuilderTest(tf.test.TestCase):

  def test_build_label_map(self):
    label_map_text_proto = """
    num_eos: 2
    character_set {
      text_string: "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ"
      delimiter: ""
    }
    """
    label_map_proto = label_map_pb2.LabelMap()
    text_format.Merge(label_map_text_proto, label_map_proto)
    label_map_object = label_map_builder.build(label_map_proto)

    test_text = tf.constant(
      ['a', 'b', '', 'abz', '0a='],
      tf.string
    )
    test_labels = label_map_object.text_to_labels(test_text)
    test_text_from_labels = label_map_object.labels_to_text(test_labels)

    with self.test_session() as sess:
      tf.tables_initializer().run()
      self.assertAllEqual(
        test_labels.eval(),
        [[3, 0, 0,  0, 0],
         [4, 0, 0,  0, 0],
         [0, 0, 0,  0, 0],
         [3, 4, 28, 0, 0],
         [2, 3, 2,  0, 0]]
      )
      self.assertAllEqual(
        test_text_from_labels.eval(),
        [b'a', b'b', b'', b'abz', b'a']
      )


if __name__ == '__main__':
  tf.test.main()
