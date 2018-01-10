import tensorflow as tf

from google.protobuf import text_format
from rare.builders import model_builder
from rare.protos import model_pb2


SINGLE_PREDICTOR_MODEL_TEXT_PROTO = """
multi_predictors_recognition_model {
  feature_extractor {
    convnet {
      crnn_net {
        net_type: SINGLE_BRANCH
        conv_hyperparams {
          op: CONV
          regularizer { l2_regularizer { weight: 1e-4 } }
          initializer { variance_scaling_initializer { } }
          batch_norm { }
        }
        summarize_activations: false
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
        initializer { variance_scaling_initializer { } }
        regularizer { l2_regularizer { weight: 1e-4 } }
      }
    }
    summarize_activations: true
  }

  predictor {
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
          built_in_set: ALLCASES
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
  }
}
"""

MULTIPLE_PREDICTOR_MODEL_TEXT_PROTO = """
multi_predictors_recognition_model {
  feature_extractor {
    convnet {
      crnn_net {
        net_type: SINGLE_BRANCH
        conv_hyperparams {
          op: CONV
          regularizer { l2_regularizer { weight: 1e-4 } }
          initializer { variance_scaling_initializer { } }
          batch_norm { }
        }
        summarize_activations: false
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
        initializer { variance_scaling_initializer { } }
        regularizer { l2_regularizer { weight: 1e-4 } }
      }
    }
    summarize_activations: true
  }

  predictor {
    name: "Forward"
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
          built_in_set: ALLCASES
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
  }

  predictor {
    name: "Backward"
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
      reverse: true
      label_map {
        character_set {
          built_in_set: ALLCASES
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
  }
}
"""

STN_MULTIPLE_PREDICTOR_MODEL_TEXT_PROTO = """
multi_predictors_recognition_model {
  spatial_transformer {
    convnet {
      stn_resnet {
        conv_hyperparams {
          op: CONV
          regularizer { l2_regularizer { } }
          initializer { variance_scaling_initializer { } }
          batch_norm { decay: 0.99 }
        }
        summarize_activations: false
      }
    }
    fc_hyperparams {
      op: CONV
      regularizer { l2_regularizer { } }
      initializer { variance_scaling_initializer { } }
      batch_norm { decay: 0.99 }
    }
    localization_h: 64
    localization_w: 128
    output_h: 32
    output_w: 100
    num_control_points: 20
    margin: 0.05
  }

  feature_extractor {
    convnet {
      crnn_net {
        net_type: SINGLE_BRANCH
        conv_hyperparams {
          op: CONV
          regularizer { l2_regularizer { weight: 1e-4 } }
          initializer { variance_scaling_initializer { } }
          batch_norm { }
        }
        summarize_activations: false
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
        initializer { variance_scaling_initializer { } }
        regularizer { l2_regularizer { weight: 1e-4 } }
      }
    }
    summarize_activations: true
  }

  predictor {
    name: "Forward"
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
          built_in_set: ALLCASES
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
  }

  predictor {
    name: "Backward"
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
      reverse: true
      label_map {
        character_set {
          built_in_set: ALLCASES
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
  }
}
"""

class ModelBuilderTest(tf.test.TestCase):

  def test_single_predictor_model_training(self):
    model_proto = model_pb2.Model()
    text_format.Merge(SINGLE_PREDICTOR_MODEL_TEXT_PROTO, model_proto)
    model_object = model_builder.build(model_proto, True)
    test_groundtruth_text_list = [
      tf.constant(b'hello', dtype=tf.string),
      tf.constant(b'world', dtype=tf.string)]
    model_object.provide_groundtruth({'groundtruth_text': test_groundtruth_text_list})
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
  
  def test_single_predictor_model_inference(self):
    model_proto = model_pb2.Model()
    text_format.Merge(SINGLE_PREDICTOR_MODEL_TEXT_PROTO, model_proto)
    model_object = model_builder.build(model_proto, False)
    test_groundtruth_text_list = [
      tf.constant(b'hello', dtype=tf.string),
      tf.constant(b'world', dtype=tf.string)]
    model_object.provide_groundtruth({'groundtruth_text': test_groundtruth_text_list})
    test_input_image = tf.random_uniform(
      shape=[2, 32, 100, 3], minval=0, maxval=255,
      dtype=tf.float32, seed=1)
    prediction_dict = model_object.predict(model_object.preprocess(test_input_image))
    recognition_dict = model_object.postprocess(prediction_dict)
    with self.test_session() as sess:
      sess.run([
        tf.global_variables_initializer(),
        tf.tables_initializer()])
      outputs = sess.run(recognition_dict)
      print(outputs)

  def test_multi_predictors_model_training(self):
    model_proto = model_pb2.Model()
    text_format.Merge(MULTIPLE_PREDICTOR_MODEL_TEXT_PROTO, model_proto)
    model_object = model_builder.build(model_proto, True)
    test_groundtruth_text_list = [
      tf.constant(b'hello', dtype=tf.string),
      tf.constant(b'world', dtype=tf.string)]
    model_object.provide_groundtruth({'groundtruth_text': test_groundtruth_text_list})
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
  
  def test_multi_predictor_model_inference(self):
    model_proto = model_pb2.Model()
    text_format.Merge(MULTIPLE_PREDICTOR_MODEL_TEXT_PROTO, model_proto)
    model_object = model_builder.build(model_proto, False)
    test_groundtruth_text_list = [
      tf.constant(b'hello', dtype=tf.string),
      tf.constant(b'world', dtype=tf.string)]
    model_object.provide_groundtruth({'groundtruth_text': test_groundtruth_text_list})
    test_input_image = tf.random_uniform(
      shape=[2, 32, 100, 3], minval=0, maxval=255,
      dtype=tf.float32, seed=1)
    prediction_dict = model_object.predict(model_object.preprocess(test_input_image))
    recognition_dict = model_object.postprocess(prediction_dict)
    with self.test_session() as sess:
      sess.run([
        tf.global_variables_initializer(),
        tf.tables_initializer()])
      outputs = sess.run(recognition_dict)
      print(outputs)

  def test_stn_multi_predictors_model_training(self):
    model_proto = model_pb2.Model()
    text_format.Merge(STN_MULTIPLE_PREDICTOR_MODEL_TEXT_PROTO, model_proto)
    model_object = model_builder.build(model_proto, True)
    test_groundtruth_text_list = [
      tf.constant(b'hello', dtype=tf.string),
      tf.constant(b'world', dtype=tf.string)]
    model_object.provide_groundtruth({'groundtruth_text': test_groundtruth_text_list})
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

  def test_stn_multi_predictor_model_inference(self):
    model_proto = model_pb2.Model()
    text_format.Merge(STN_MULTIPLE_PREDICTOR_MODEL_TEXT_PROTO, model_proto)
    model_object = model_builder.build(model_proto, False)
    test_groundtruth_text_list = [
      tf.constant(b'hello', dtype=tf.string),
      tf.constant(b'world', dtype=tf.string)]
    model_object.provide_groundtruth({'groundtruth_text': test_groundtruth_text_list})
    test_input_image = tf.random_uniform(
      shape=[2, 32, 100, 3], minval=0, maxval=255,
      dtype=tf.float32, seed=1)
    prediction_dict = model_object.predict(model_object.preprocess(test_input_image))
    recognition_dict = model_object.postprocess(prediction_dict)
    with self.test_session() as sess:
      sess.run([
        tf.global_variables_initializer(),
        tf.tables_initializer()])
      outputs = sess.run(recognition_dict)
      print(outputs)

if __name__ == '__main__':
  tf.test.main()
