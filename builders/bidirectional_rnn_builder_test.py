import tensorflow as tf

from google.protobuf import text_format
from aster.builders import bidirectional_rnn_builder
from aster.protos import bidirectional_rnn_pb2


class BidirectionalRnnBuilderTest(tf.test.TestCase):

  def test_bidirectional_rnn(self):
    text_proto = """
    static: true
    fw_bw_rnn_cell {
      lstm_cell {
        num_units: 32
        forget_bias: 1.0
        initializer { orthogonal_initializer {} }
      }
    }
    rnn_regularizer { l2_regularizer { weight: 1e-4 } }
    num_output_units: 31
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

    test_input = tf.random_uniform([2, 5, 32], dtype=tf.float32)
    test_output = brnn_object.predict(test_input)

    with self.test_session() as sess:
      tf.global_variables_initializer().run()
      sess_outputs = sess.run({'outputs': test_output})
      self.assertAllEqual(sess_outputs['outputs'].shape, [2, 5, 31])

  def test_dynamic_bidirectional_rnn(self):
    text_proto = """
    static: false
    fw_bw_rnn_cell {
      lstm_cell {
        num_units: 32
        forget_bias: 1.0
        initializer { orthogonal_initializer {} }
      }
    }
    rnn_regularizer { l2_regularizer { weight: 1e-4 } }
    num_output_units: 31
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

    test_input = tf.random_uniform([2, 5, 32], dtype=tf.float32)
    test_output = brnn_object.predict(test_input)

    with self.test_session() as sess:
      tf.global_variables_initializer().run()
      sess_outputs = sess.run({'outputs': test_output})
      self.assertAllEqual(sess_outputs['outputs'].shape, [2, 5, 31])

  def test_bidirectional_rnn_nofc(self):
    text_proto = """
    static: true
    fw_bw_rnn_cell {
      lstm_cell {
        num_units: 32
        forget_bias: 1.0
        initializer { orthogonal_initializer {} }
      }
    }
    rnn_regularizer { l2_regularizer { weight: 1e-4 } }
    """
    config = bidirectional_rnn_pb2.BidirectionalRnn()
    text_format.Merge(text_proto, config)
    brnn_object = bidirectional_rnn_builder.build(config, True)

    test_input = tf.random_uniform([2, 5, 32], dtype=tf.float32)
    test_output = brnn_object.predict(test_input)

    with self.test_session() as sess:
      tf.global_variables_initializer().run()
      sess_outputs = sess.run({'outputs': test_output})
      self.assertAllEqual(sess_outputs['outputs'].shape, [2, 5, 64])

if __name__ == '__main__':
  tf.test.main()
