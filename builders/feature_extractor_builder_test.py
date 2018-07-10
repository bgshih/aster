import tensorflow as tf
from google.protobuf import text_format

from aster.protos import feature_extractor_pb2
from aster.builders import feature_extractor_builder


class FeatureExtractorBuilderTest(tf.test.TestCase):

  def test_feature_extractor_builder_single_branch(self):
    text_proto = """
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
    """
    feature_extractor_proto = feature_extractor_pb2.FeatureExtractor()
    text_format.Merge(text_proto, feature_extractor_proto)
    feature_extractor_object = feature_extractor_builder.build(feature_extractor_proto, True)

    test_image_shape = [2, 32, 128, 3]
    test_input_image = tf.random_uniform(
      test_image_shape,
      minval=0,
      maxval=255.0,
      dtype=tf.float32,
      seed=1
    )
    feature_maps = feature_extractor_object.extract_features(test_input_image)
    print('Outputs of test_feature_extractor_builder: {}'.format(feature_maps))

  def test_feature_extractor_builder_multi_branches(self):
    text_proto = """
    convnet {
      crnn_net {
        net_type: THREE_BRANCHES
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
    """
    feature_extractor_proto = feature_extractor_pb2.FeatureExtractor()
    text_format.Merge(text_proto, feature_extractor_proto)
    feature_extractor_object = feature_extractor_builder.build(feature_extractor_proto, True)

    test_image_shape = [2, 32, 128, 3]
    test_input_image = tf.random_uniform(
      test_image_shape,
      minval=0,
      maxval=255.0,
      dtype=tf.float32,
      seed=1
    )
    feature_maps = feature_extractor_object.extract_features(test_input_image)
    self.assertTrue(len(feature_maps) == 3)
    print('Outputs of test_feature_extractor_builder_multi_branches: {}'.format(feature_maps))

if __name__ == '__main__':
  tf.test.main()
