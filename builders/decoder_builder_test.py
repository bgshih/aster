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
          num_units: 256
        }
      }
      num_attention_units: 256
      attention_conv_kernel_size: 5
    }
    """
    decoder_proto = decoder_pb2.Decoder()
    text_format.Merge(decoder_proto, decoder_proto)
    decoder_object = decoder_builder.build(decoder_proto)

    self.assertEqual(decoder_object.num_attention_units, 256)
    self.assertEqual(decoder_object.attention_conv_kernel_size, 5)
