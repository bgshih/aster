import tensorflow as tf

from google.protobuf import text_format
from rare.core import rnn_cell, rnn_cell_pb2


class RnnCellTest(tf.test.TestCase):

  def test_build_lstm_cell(self):
    rnn_cell_text_proto = """
    lstm_cell {
      num_units: 1024
      use_peepholes: true
      forget_bias: 1.5
      initializer { orthogonal_initializer { seed: 1 } }
    }
    """
    rnn_cell_proto = rnn_cell_pb2.RnnCell()
    text_format.Merge(rnn_cell_text_proto, rnn_cell_proto)
    rnn_cell_object = rnn_cell.build(rnn_cell_proto)

    lstm_state_tuple = rnn_cell_object.state_size

    self.assertEqual(lstm_state_tuple[0], 1024)
    self.assertEqual(lstm_state_tuple[1], 1024)

  def test_build_gru_cell(self):
    rnn_cell_text_proto = """
    gru_cell {
      num_units: 1024
      initializer { orthogonal_initializer { seed: 1 } }
    }
    """
    rnn_cell_proto = rnn_cell_pb2.RnnCell()
    text_format.Merge(rnn_cell_text_proto, rnn_cell_proto)
    rnn_cell_object = rnn_cell.build(rnn_cell_proto)

    self.assertEqual(rnn_cell_object.state_size, 1024)

if __name__ == '__main__':
  tf.test.main()
