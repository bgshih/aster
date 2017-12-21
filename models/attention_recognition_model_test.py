import tensorflow as tf

from google.protobuf import text_format
from rare.models import model, model_pb2


class AttentionRecognitionModdelTest(tf.test.TestCase):

  def test_build_attention_model(self):
    model_text_proto = """
    attention_recognition_model {
      feature_extractor {
        baseline_feature_extractor {
          conv_hyperparams {
            op: CONV
            initializer { variance_scaling_initializer {} }
            regularizer { l2_regularizer {weight: 0 } }
            batch_norm { }
          }
          summarize_inputs: false
        }
      }

      bidirectional_rnn {
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
          initializer { variance_scaling_initializer {} }
          regularizer { l2_regularizer { weight: 1e-4 } }
        }
      }

      bidirectional_rnn {
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
          initializer { variance_scaling_initializer {} }
          regularizer { l2_regularizer { weight: 1e-4 } }
        }
      }

      attention_predictor {
        bahdanau_attention_predictor {
          rnn_cell {
            lstm_cell {
              num_units: 256
              forget_bias: 1.0
              initializer { orthogonal_initializer { } }
            }
          }
          num_attention_units: 128
          max_num_steps: 10
          rnn_regularizer { l2_regularizer { weight: 1e-4 } }
          fc_hyperparams {
            op: FC
            activation: RELU
            initializer { variance_scaling_initializer {} }
            regularizer { l2_regularizer { weight: 1e-4 } }
          }
        }
      }

      label_map {
        character_set {
          text_string: "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ"
          delimiter: ""
        }
        label_offset: 2
      }
      
      loss {
        sequence_cross_entropy_loss {
          sequence_normalize: false
          sample_normalize: true
        }
      }
    }
    """
    model_proto = model_pb2.Model()
    text_format.Merge(model_text_proto, model_proto)
    model_object = model.build(model_proto, True)

    test_groundtruth_text_list = [
      tf.constant(b'hello', dtype=tf.string),
      tf.constant(b'world', dtype=tf.string)]
    model_object.provide_groundtruth(test_groundtruth_text_list)
    test_input_image = tf.random_uniform(
      shape=[2, 32, 100, 3], minval=0, maxval=255,
      dtype=tf.float32, seed=1)
    prediction_dict = model_object.predict(model_object.preprocess(test_input_image))
    loss = model_object.loss(prediction_dict)

    with self.test_session() as sess:
      sess.run([
        tf.global_variables_initializer(),
        tf.tables_initializer()])
      outputs = sess.run({'loss': loss})
      print(outputs['loss'])

  # def test_attention_predictor(self):
  #   rnn_cell_object = tf.contrib.rnn.GRUCell(256)
  #   label_map_object = label_map.LabelMap(character_set=list(string.ascii_lowercase))

  #   predictor_object = attention_predictor.BahdanauAttentionPredictor(
  #     rnn_cell_object,
  #     num_attention_units=256,
  #     max_num_steps=50,
  #     is_training=True)

  #   test_feature_map = tf.random_uniform([2, 1, 15, 32], minval=0, maxval=3, seed=2)
  #   test_num_steps = tf.constant(4, dtype=tf.int32)
  #   test_groundtruth_text = ['ab', 'abcd']
  #   decoder_inputs, decoder_inputs_lenghts = label_map_object.text_to_labels(test_groundtruth_text, return_lengths=True)

  #   output_logits, output_labels, output_lengths = predictor_object.predict(
  #     test_feature_map,
  #     decoder_inputs=decoder_inputs,
  #     decoder_inputs_lengths=decoder_inputs_lenghts,
  #     num_classes=label_map_object.num_classes,
  #     go_label=label_map_object.go_label,
  #     eos_label=label_map_object.eos_label)

  #   with self.test_session() as sess:
  #     sess.run([
  #       tf.global_variables_initializer(),
  #       tf.tables_initializer()])
  #     outputs = sess.run({
  #       'logits': output_logits,
  #       'labels': output_labels,
  #       'lengths': output_lengths
  #     })
  #     print(outputs)

if __name__ == '__main__':
  tf.test.main()
