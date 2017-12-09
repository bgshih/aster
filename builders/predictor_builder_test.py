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
    }
    """
    predictor_proto = predictor_pb2.Predictor()
    text_format.Merge(predictor_text_proto, predictor_proto)

    test_label_map = label_map.LabelMap(
      character_set=list(string.ascii_lowercase),
      num_eos=1)
    predictor_object = predictor_builder.build(
      predictor_proto,
      label_map=test_label_map,
      is_training=True
    )

    # self.assertEqual(predictor_object._num_attention_units, 256)
    # self.assertEqual(predictor_object._attention_conv_kernel_size, 5)

    test_batch_size = 1
    test_num_steps = 2
    test_num_labels = 2
    test_feature_map = tf.constant([[[[1], [-1]], [[2], [-2]]]], dtype=tf.float32)
    test_decoder_inputs = tf.constant([[1, 0]])
    # logits = predictor_object.decode(
    #   test_feature_map,
    #   test_num_steps,
    #   test_num_labels,
    #   test_predictor_inputs
    # )

    logits = predictor_object.predict(
      test_feature_map,
      decoder_inputs=test_decoder_inputs,
      decoder_inputs_lengths=[1])

    # with self.test_session() as sess:
    #   sess.run(tf.global_variables_initializer())
    #   outputs_dict = sess.run({'logits': logits})
    #   print(outputs_dict['logits'])


if __name__ == '__main__':
  tf.test.main()
