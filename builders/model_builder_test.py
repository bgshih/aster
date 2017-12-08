import tensorflow as tf

from google.protobuf import text_format
from rare.builders import model_builder
from rare.protos import model_pb2


def ModelBuilderTest(tf.test.TestCase):

  def test_build_model(self):
    model_text_proto = """
    attention_recognition_model {
      num_classes: 27
      feature_extractor {
        baseline_feature_extractor {
        }
      }
      label_map {

      }
      loss {
        sequence_cross_entropy_loss {
          sequence_normalize: true
          sample_normalize: true
        }
      }
    }
    """
    model_proto = model_pb2.RecognitionModel()
    text_format.Merge(model_text_proto, model_proto)
    model_object = model_builder.build(model_proto, True)


if __name__ == '__main__':
  tf.test.main()
