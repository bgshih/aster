import os
import numpy as np
import tensorflow as tf

from google.protobuf import text_format

from tensorflow.core.example import example_pb2
from tensorflow.core.example import feature_pb2
from rare.builders import input_reader_builder
from rare.core import standard_fields as fields
from rare.protos import input_reader_pb2


class InputReaderBuilderTest(tf.test.TestCase):

  def create_tf_record(self):
    path = os.path.join(self.get_temp_dir(), 'tfrecord')
    writer = tf.python_io.TFRecordWriter(path)

    image_tensor = np.random.randint(255, size=(4, 5, 3)).astype(np.uint8)
    with self.test_session():
      encoded_jpeg = tf.image.encode_jpeg(tf.constant(image_tensor)).eval()
    example = example_pb2.Example(features=feature_pb2.Features(feature={
        'image/encoded': feature_pb2.Feature(
            bytes_list=feature_pb2.BytesList(value=[encoded_jpeg])),
        'image/format': feature_pb2.Feature(
            bytes_list=feature_pb2.BytesList(value=['jpeg'.encode('utf-8')])),
        'image/transcript': feature_pb2.Feature(
            bytes_list=feature_pb2.BytesList(value=[
                'hello'.encode('utf-8')]))
    }))
    writer.write(example.SerializeToString())
    writer.close()

    return path

  def test_build_tf_record_input_reader(self):
    tf_record_path = self.create_tf_record()

    input_reader_text_proto = """
      shuffle: false
      num_readers: 3
      tf_record_input_reader {{
        input_path: '{0}'
      }}
    """.format(tf_record_path)
    input_reader_proto = input_reader_pb2.InputReader()
    text_format.Merge(input_reader_text_proto, input_reader_proto)
    tensor_dict = input_reader_builder.build(input_reader_proto)

    sv = tf.train.Supervisor(logdir=self.get_temp_dir())
    with sv.prepare_or_wait_for_session() as sess:
      sv.start_queue_runners(sess)
      output_dict = sess.run(tensor_dict)

    self.assertEqual(
        (4, 5, 3),
        output_dict[fields.InputDataFields.image].shape)
    self.assertEqual(
        'hello'.encode('utf-8'),
        output_dict[fields.InputDataFields.groundtruth_text])


if __name__ == '__main__':
  tf.test.main()
