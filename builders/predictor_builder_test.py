import string

import tensorflow as tf

from google.protobuf import text_format
from rare.builders import predictor_builder
from rare.protos import predictor_pb2
from rare.core import label_map


class PredictorBuilderTest(tf.test.TestCase):

  def test_build_predictor(self):
    predictor_text_proto = """
    attention_predictor {
      rnn_cell {
        gru_cell {
          num_units: 512
        }
      }
      num_attention_units: 128
      attention_conv_kernel_size: 5
      output_embedding {
        one_hot_embedding {
        }
      }
      max_num_steps: 10
    }
    """
    predictor_proto = predictor_pb2.Predictor()
    text_format.Merge(predictor_text_proto, predictor_proto)
    predictor_object = predictor_builder.build(predictor_proto, True)
    
    test_batch_size = 1
    test_num_steps = tf.constant(2, tf.int32)
    test_num_classes = 2
    test_feature_map = tf.constant([[[[1], [-1]], [[2], [-2]]]], dtype=tf.float32)
    test_decoder_inputs = tf.constant([[1, 0]])
    test_logits = predictor_object.predict(
      test_feature_map,
      test_num_steps,
      num_classes=test_num_classes,
      decoder_inputs=test_decoder_inputs
    )

    with self.test_session() as sess:
      sess.run(tf.global_variables_initializer())
      outputs_dict = sess.run({'logits': test_logits})
      print(outputs_dict['logits'])


if __name__ == '__main__':
  tf.test.main()
