import string

import tensorflow as tf

from google.protobuf import text_format
from rare.builders import predictor_builder
from rare.protos import predictor_pb2
from rare.core import label_map


class PredictorBuilderTest(tf.test.TestCase):

  def test_build_predictor(self):
    predictor_text_proto = """
    bahdanau_attention_predictor {
      rnn_cell {
        gru_cell {
          num_units: 512
        }
      }
      num_attention_units: 128
      max_num_steps: 40
    }
    """
    predictor_proto = predictor_pb2.Predictor()
    text_format.Merge(predictor_text_proto, predictor_proto)
    predictor_object = predictor_builder.build(predictor_proto, False)

    self.assertEqual(predictor_object._num_attention_units, 128)
    self.assertEqual(predictor_object._max_num_steps, 40)
    self.assertEqual(predictor_object._is_training, False)


if __name__ == '__main__':
  tf.test.main()
