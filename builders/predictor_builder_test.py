import tensorflow as tf
from google.protobuf import text_format

from rare.builders import predictor_builder
from rare.protos import predictor_pb2


class PredictorBuilderTest(tf.test.TestCase):

  def test_predictor_builder(self):
    predictor_text_proto = """
    attention_predictor {
      rnn_cell {
        lstm_cell {
          num_units: 256
          forget_bias: 1.0
          initializer { orthogonal_initializer { } }
        }
      }
      rnn_regularizer { l2_regularizer { weight: 1e-4 } }
      num_attention_units: 128
      max_num_steps: 10
      multi_attention: false
      beam_width: 1
      reverse: false
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
    predictor_proto = predictor_pb2.Predictor()
    text_format.Merge(predictor_text_proto, predictor_proto)
    predictor_object = predictor_builder.build(predictor_proto, True)

    feature_maps = [tf.random_uniform([2, 1, 10, 32], dtype=tf.float32)]
    predictor_object.provide_groundtruth(
      tf.constant([b'hello', b'world'], dtype=tf.string)
    )
    predictions_dict = predictor_object.predict(feature_maps)
    loss = predictor_object.loss(predictions_dict)

    with self.test_session() as sess:
      sess.run([
        tf.global_variables_initializer(),
        tf.tables_initializer()])
      sess_outputs = sess.run({'loss': loss})
      print(sess_outputs)

  def test_predictor_with_lm_builder(self):
    predictor_text_proto = """
    attention_predictor {
      rnn_cell {
        lstm_cell {
          num_units: 256
          forget_bias: 1.0
          initializer { orthogonal_initializer { } }
        }
      }
      rnn_regularizer { l2_regularizer { weight: 1e-4 } }
      num_attention_units: 128
      max_num_steps: 10
      multi_attention: false
      beam_width: 1
      reverse: false
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
      lm_rnn_cell {
        rnn_cell {
          lstm_cell {
            num_units: 256
            forget_bias: 1.0
            initializer { orthogonal_initializer { } }
          }
        }
        rnn_cell {
          lstm_cell {
            num_units: 256
            forget_bias: 1.0
            initializer { orthogonal_initializer { } }
          }
        }
      }
    }
    """
    predictor_proto = predictor_pb2.Predictor()
    text_format.Merge(predictor_text_proto, predictor_proto)
    predictor_object = predictor_builder.build(predictor_proto, True)

    feature_maps = [tf.random_uniform([2, 1, 10, 32], dtype=tf.float32)]
    predictor_object.provide_groundtruth(
      tf.constant([b'hello', b'world'], dtype=tf.string)
    )
    predictions_dict = predictor_object.predict(feature_maps)
    loss = predictor_object.loss(predictions_dict)

    with self.test_session() as sess:
      sess.run([
        tf.global_variables_initializer(),
        tf.tables_initializer()])
      sess_outputs = sess.run({'loss': loss})
      print(sess_outputs)
  
  # def test_sync_predictor_builder(self):
  #   predictor_text_proto = """
  #   attention_predictor {
  #     rnn_cell {
  #       lstm_cell {
  #         num_units: 256
  #         forget_bias: 1.0
  #         initializer { orthogonal_initializer { } }
  #       }
  #     }
  #     rnn_regularizer { l2_regularizer { weight: 1e-4 } }
  #     num_attention_units: 128
  #     max_num_steps: 10
  #     multi_attention: false
  #     beam_width: 1
  #     reverse: false
  #     label_map {
  #       character_set {
  #         text_string: "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ"
  #         delimiter: ""
  #       }
  #       label_offset: 2
  #     }
  #     loss {
  #       sequence_cross_entropy_loss {
  #         sequence_normalize: false
  #         sample_normalize: true
  #       }
  #     }
  #     sync: true
  #   }
  #   """
  #   predictor_proto = predictor_pb2.Predictor()
  #   text_format.Merge(predictor_text_proto, predictor_proto)
  #   predictor_object = predictor_builder.build(predictor_proto, True)

  #   feature_maps = [tf.random_uniform([2, 1, 10, 32], dtype=tf.float32)]
  #   predictor_object.provide_groundtruth(
  #     tf.constant([b'hello', b'world'], dtype=tf.string)
  #   )
  #   predictions_dict = predictor_object.predict(feature_maps)
  #   loss = predictor_object.loss(predictions_dict)

  #   with self.test_session() as sess:
  #     sess.run([
  #       tf.global_variables_initializer(),
  #       tf.tables_initializer()])
  #     sess_outputs = sess.run({'loss': loss})
  #     print(sess_outputs)

if __name__ == '__main__':
  tf.test.main()
