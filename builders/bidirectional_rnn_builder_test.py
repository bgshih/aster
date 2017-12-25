import tensorflow as tf

from google.protobuf import text_format
from rare.builders import bidirectional_rnn_builder
from rare.protos import bidirectional_rnn_pb2


class BidirectionalRnnBuilderTest(tf.test.TestCase):

  def test_bidirectional_rnn(self):
    text_proto = """
    fw_bw_rnn_cell {
      lstm_cell {
        num_units: 256
        forget_bias: 1.0
        initializer { orthogonal_initializer {} }
      }
    }
    rnn_regularizer { l2_regularizer { weight: 1e-4 } }
    num_output_units: 256
    fc_hyperparams {
      op: FC
      activation: RELU
      initializer { variance_scaling_initializer { } }
      regularizer { l2_regularizer { weight: 1e-4 } }
    }
    """
    config = bidirectional_rnn_pb2.BidirectionalRnn()
    text_format.Merge(text_proto, config)
    brnn_object = bidirectional_rnn_builder.build(config, True)

  def test_bidirectional_rnn_nofc(self):
    text_proto = """
    fw_bw_rnn_cell {
      lstm_cell {
        num_units: 256
        forget_bias: 1.0
        initializer { orthogonal_initializer {} }
      }
    }
    rnn_regularizer { l2_regularizer { weight: 1e-4 } }
    """
    config = bidirectional_rnn_pb2.BidirectionalRnn()
    text_format.Merge(text_proto, config)
    brnn_object = bidirectional_rnn_builder.build(config, True)

if __name__ == '__main__':
  tf.test.main()
