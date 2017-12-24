import tensorflow as tf

from google.protobuf import text_format
from rare.core import feature_extractor, feature_extractor_pb2
from rare.feature_extractors import baseline_feature_extractor, resnet_feature_extractor


class FeatureExtractorTest(tf.test.TestCase):

  def test_baseline_feature_extractor(self):
    feature_extractor_text_proto = """
    baseline_feature_extractor {
      baseline_type: SINGLE_BRANCH
      conv_hyperparams {
        op: CONV
        regularizer {
          l2_regularizer {
            weight: 1e-4
          }
        }
        initializer {
          variance_scaling_initializer {
          }
        }
        batch_norm {
        }
      }
    }
    """
    feature_extractor_proto = feature_extractor_pb2.FeatureExtractor()
    text_format.Merge(feature_extractor_text_proto, feature_extractor_proto)
    feature_extractor_object = feature_extractor.build(feature_extractor_proto, True)
    self.assertTrue(
      isinstance(feature_extractor_object, baseline_feature_extractor.BaselineFeatureExtractor))

    test_image_shape = [2, 32, 128, 3]
    test_input_image = tf.random_uniform(
      test_image_shape,
      minval=0,
      maxval=255.0,
      dtype=tf.float32,
      seed=1
    )
    feature_maps = feature_extractor_object.extract_features(test_input_image)
    self.assertTrue(len(feature_maps) == 1)
    print(feature_maps)

  def test_baseline_feature_extractor_two_branch(self):
    feature_extractor_text_proto = """
    baseline_feature_extractor {
      baseline_type: TWO_BRANCH
      conv_hyperparams {
        op: CONV
        regularizer {
          l2_regularizer {
            weight: 1e-4
          }
        }
        initializer {
          variance_scaling_initializer {
          }
        }
        batch_norm {
        }
      }
    }
    """
    feature_extractor_proto = feature_extractor_pb2.FeatureExtractor()
    text_format.Merge(feature_extractor_text_proto, feature_extractor_proto)
    feature_extractor_object = feature_extractor.build(feature_extractor_proto, True)
    self.assertTrue(
      isinstance(feature_extractor_object, baseline_feature_extractor.BaselineTwoBranchFeatureExtractor))

    test_image_shape = [2, 32, 128, 3]
    test_input_image = tf.random_uniform(
      test_image_shape,
      minval=0,
      maxval=255.0,
      dtype=tf.float32,
      seed=1
    )
    feature_maps = feature_extractor_object.extract_features(test_input_image)
    self.assertTrue(len(feature_maps) == 2)
    print(feature_maps)

  def test_baseline_feature_extractor_three_branch(self):
    feature_extractor_text_proto = """
    baseline_feature_extractor {
      baseline_type: THREE_BRANCH
      conv_hyperparams {
        op: CONV
        regularizer {
          l2_regularizer {
            weight: 1e-4
          }
        }
        initializer {
          variance_scaling_initializer {
          }
        }
        batch_norm {
        }
      }
    }
    """
    feature_extractor_proto = feature_extractor_pb2.FeatureExtractor()
    text_format.Merge(feature_extractor_text_proto, feature_extractor_proto)
    feature_extractor_object = feature_extractor.build(feature_extractor_proto, True)
    self.assertTrue(
      isinstance(feature_extractor_object, baseline_feature_extractor.BaselineThreeBranchFeatureExtractor))

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
    print(feature_maps)

  def test_baseline_feature_extractor_three_branch_with_brnn(self):
    feature_extractor_text_proto = """
    baseline_feature_extractor {
      baseline_type: THREE_BRANCH
      conv_hyperparams {
        op: CONV
        regularizer {
          l2_regularizer {
            weight: 1e-4
          }
        }
        initializer {
          variance_scaling_initializer {
          }
        }
        batch_norm {
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
        rnn_regularizer { l2_regularizer { weight: 0 } }
        num_output_units: 256
        fc_hyperparams {
          op: FC
          activation: RELU
          initializer { variance_scaling_initializer {} }
          regularizer { l2_regularizer { weight: 0 } }
        }
      }
    }
    """
    feature_extractor_proto = feature_extractor_pb2.FeatureExtractor()
    text_format.Merge(feature_extractor_text_proto, feature_extractor_proto)
    feature_extractor_object = feature_extractor.build(feature_extractor_proto, True)
    self.assertTrue(
      isinstance(feature_extractor_object, baseline_feature_extractor.BaselineThreeBranchFeatureExtractor))

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
    print(feature_maps)

  def test_resnet50_feature_extractor(self):
    feature_extractor_text_proto = """
    resnet_feature_extractor {
      resnet_type: RESNET_50

      conv_hyperparams {
        op: CONV
        regularizer {
          l2_regularizer {
            weight: 1e-4
          }
        }
        initializer {
          variance_scaling_initializer {
          }
        }
        batch_norm {
          decay: 0.99
        }
      }

      summarize_inputs: false
    }
    """
    feature_extractor_proto = feature_extractor_pb2.FeatureExtractor()
    text_format.Merge(feature_extractor_text_proto, feature_extractor_proto)
    feature_extractor_object = feature_extractor.build(feature_extractor_proto, True)

    test_image_shape = [2, 32, 128, 3]
    test_input_image = tf.random_uniform(
      test_image_shape,
      minval=0,
      maxval=255.0,
      dtype=tf.float32,
      seed=1
    )
    feature_maps = feature_extractor_object.extract_features(test_input_image)
    print(feature_maps)

if __name__ == '__main__':
  tf.test.main()
