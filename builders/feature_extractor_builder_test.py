import tensorflow as tf

from google.protobuf import text_format
from rare.builders import feature_extractor_builder
from rare.protos import feature_extractor_pb2

class FeatureExtractorBuilderTest(tf.test.TestCase):

  def test_baseline_feature_extractor(self):
    feature_extractor_text_proto = """
    baseline_feature_extractor {
    }
    """
    feature_extractor_proto = feature_extractor_pb2.FeatureExtractor()
    text_format.Merge(feature_extractor_text_proto, feature_extractor_proto)
    feature_extractor_object = feature_extractor_builder.build(feature_extractor_proto)

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
