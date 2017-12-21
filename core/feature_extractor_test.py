import tensorflow as tf

from google.protobuf import text_format
from rare.core import feature_extractor, feature_extractor_pb2


class FeatureExtractorTest(tf.test.TestCase):

  def test_baseline_feature_extractor(self):
    feature_extractor_text_proto = """
    baseline_feature_extractor {
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
