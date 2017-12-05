import tensorflow as tf

from google.protobuf import text_format
from rare.builders import decoder_builder
from rare.protos import decoder_pb2


class DecoderBuilderTest(tf.test.TestCase):

  def test_build_decoder(self):
    decoder_text_proto = """
    attention_decoder {
      rnn_cell {
        gru_cell {
          num_units: 512
        }
      }
      num_attention_units: 256
      attention_conv_kernel_size: 5
      output_embedding {
        one_hot_embedding {
        }
      }
    }
    """
    decoder_proto = decoder_pb2.Decoder()
    text_format.Merge(decoder_text_proto, decoder_proto)
    decoder_object = decoder_builder.build(decoder_proto, True)

    self.assertEqual(decoder_object._num_attention_units, 256)
    self.assertEqual(decoder_object._attention_conv_kernel_size, 5)

    test_batch_size = 1
    test_num_steps = 2
    test_num_labels = 2
    test_feature_map = tf.constant([[[[1], [-1]], [[2], [-2]]]], dtype=tf.float32)
    test_decoder_inputs = tf.constant([[1, 0]])
    logits = decoder_object.decode(
      test_feature_map,
      test_num_steps,
      test_num_labels,
      test_decoder_inputs
    )

    with self.test_session() as sess:
      sess.run(tf.global_variables_initializer())
      outputs_dict = sess.run({'logits': logits})
      print(outputs_dict['logits'])

      # save graph
      writer = tf.summary.FileWriter('/tmp/rare_tests/', sess.graph)


if __name__ == '__main__':
  tf.test.main()
