import tensorflow as tf

from google.protobuf import text_format
from rare.builders import model_builder
from rare.protos import model_pb2


class ModelBuilderTest(tf.test.TestCase):

  def test_build_attention_model(self):
    model_text_proto = """
    attention_recognition_model {
      feature_extractor {
        baseline_feature_extractor {
          conv_hyperparams {
            op: CONV
            initializer {
              variance_scaling_initializer {
              }
            }
            regularizer {
              l2_regularizer {
                weight: 0
              }
            }
            batch_norm {
            }
          }
          summarize_inputs: false
        }
      }

      bidirectional_rnn_cell {
        gru_cell {
          num_units: 256
        }
      }

      bidirectional_rnn_cell {
        gru_cell {
          num_units: 256
        }
      }

      predictor {
        bahdanau_attention_predictor {
          rnn_cell {
            gru_cell {
              num_units: 512
            }
          }
          num_attention_units: 128
          max_num_steps: 10
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
    model_proto = model_pb2.RecognitionModel()
    text_format.Merge(model_text_proto, model_proto)
    model_object = model_builder.build(model_proto, True)

    test_groundtruth_text_list = [
      tf.constant(b'hello', dtype=tf.string),
      tf.constant(b'world', dtype=tf.string)]
    model_object.provide_groundtruth(test_groundtruth_text_list)
    test_input_image = tf.random_uniform(
      shape=[2, 32, 100, 3],
      minval=0,
      maxval=255,
      dtype=tf.float32,
      seed=1
    )
    prediction_dict = model_object.predict(model_object.preprocess(test_input_image))
    loss = model_object.loss(prediction_dict)

    with self.test_session() as sess:
      sess.run([
        tf.global_variables_initializer(),
        tf.tables_initializer()])
      outputs = sess.run({'loss': loss})
      print(outputs['loss'])


  def test_build_ctc_model(self):
    model_text_proto = """
    ctc_recognition_model {
      feature_extractor {
        baseline_feature_extractor {
          conv_hyperparams {
            op: CONV
            initializer {
              variance_scaling_initializer {
              }
            }
            regularizer {
              l2_regularizer {
                weight: 0
              }
            }
            batch_norm {
            }
          }
          summarize_inputs: false
        }
      }

      bidirectional_rnn_cell {
        gru_cell {
          num_units: 256
        }
      }

      bidirectional_rnn_cell {
        gru_cell {
          num_units: 256
        }
      }

      label_map {
        character_set {
          text_string: "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ"
          delimiter: ""
        }
        label_offset: 0
      }
    }
    """
    model_proto = model_pb2.RecognitionModel()
    text_format.Merge(model_text_proto, model_proto)
    model_object = model_builder.build(model_proto, True)

    test_groundtruth_text_list = [
      tf.constant(b'hello', dtype=tf.string),
      tf.constant(b'world', dtype=tf.string)]
    model_object.provide_groundtruth(test_groundtruth_text_list)
    test_input_image = tf.random_uniform(
      shape=[2, 32, 100, 3],
      minval=0,
      maxval=255,
      dtype=tf.float32,
      seed=1
    )
    prediction_dict = model_object.predict(model_object.preprocess(test_input_image))
    loss = model_object.loss(prediction_dict)

    with self.test_session() as sess:
      sess.run([
        tf.global_variables_initializer(),
        tf.tables_initializer()])
      outputs = sess.run({'loss': loss})
      print(outputs['loss'])


if __name__ == '__main__':
  tf.test.main()
