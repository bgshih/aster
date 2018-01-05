import tensorflow as tf
from google.protobuf import text_format

from rare.builders import convnet_builder
from rare.protos import convnet_pb2
from rare.convnets import crnn_net
from rare.convnets import resnet

class FeatureExtractorTest(tf.test.TestCase):

  def test_crnn_net_single_branch(self):
    feature_extractor_text_proto = """
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
    """
    convnet_proto = convnet_pb2.Convnet()
    text_format.Merge(feature_extractor_text_proto, convnet_proto)
    convnet_object = convnet_builder.build(convnet_proto, True)
    self.assertTrue(
      isinstance(convnet_object, crnn_net.CrnnNet))

    test_image_shape = [2, 32, 128, 3]
    test_input_image = tf.random_uniform(
      test_image_shape,
      minval=0,
      maxval=255.0,
      dtype=tf.float32,
      seed=1
    )
    feature_maps = convnet_object.extract_features(test_input_image)
    self.assertTrue(len(feature_maps) == 1)
    print('Outputs of test_crnn_net_single_branch: {}'.format(feature_maps))
  
  def test_crnn_net_two_branches(self):
    feature_extractor_text_proto = """
    crnn_net {
      net_type: TWO_BRANCHES
      conv_hyperparams {
        op: CONV
        regularizer { l2_regularizer { weight: 1e-4 } }
        initializer { variance_scaling_initializer { } }
        batch_norm { }
      }
      summarize_activations: false
    }
    """
    convnet_proto = convnet_pb2.Convnet()
    text_format.Merge(feature_extractor_text_proto, convnet_proto)
    convnet_object = convnet_builder.build(convnet_proto, True)
    self.assertTrue(
      isinstance(convnet_object, crnn_net.CrnnNet))

    test_image_shape = [2, 32, 128, 3]
    test_input_image = tf.random_uniform(
      test_image_shape,
      minval=0,
      maxval=255.0,
      dtype=tf.float32,
      seed=1
    )
    feature_maps = convnet_object.extract_features(test_input_image)
    self.assertTrue(len(feature_maps) == 2)
    print('Outputs of test_crnn_net_two_branches: {}'.format(feature_maps))
  
  def test_crnn_net_three_branches(self):
    feature_extractor_text_proto = """
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
    """
    convnet_proto = convnet_pb2.Convnet()
    text_format.Merge(feature_extractor_text_proto, convnet_proto)
    convnet_object = convnet_builder.build(convnet_proto, True)
    self.assertTrue(
      isinstance(convnet_object, crnn_net.CrnnNet))

    test_image_shape = [2, 32, 128, 3]
    test_input_image = tf.random_uniform(
      test_image_shape,
      minval=0,
      maxval=255.0,
      dtype=tf.float32,
      seed=1
    )
    feature_maps = convnet_object.extract_features(test_input_image)
    self.assertTrue(len(feature_maps) == 3)
    print('Outputs of test_crnn_net_three_branches: {}'.format(feature_maps))
  
  def test_resnet_50layer(self):
    feature_extractor_text_proto = """
    resnet {
      net_type: SINGLE_BRANCH
      net_depth: RESNET_50
      conv_hyperparams {
        op: CONV
        regularizer { l2_regularizer { weight: 1e-4 } }
        initializer { variance_scaling_initializer { } }
        batch_norm { }
      }
      summarize_activations: false
    }
    """
    convnet_proto = convnet_pb2.Convnet()
    text_format.Merge(feature_extractor_text_proto, convnet_proto)
    convnet_object = convnet_builder.build(convnet_proto, True)
    self.assertTrue(
      isinstance(convnet_object, resnet.Resnet50Layer))
    test_image_shape = [2, 32, 128, 3]
    test_input_image = tf.random_uniform(
      test_image_shape,
      minval=0,
      maxval=255.0,
      dtype=tf.float32,
      seed=1
    )
    feature_maps = convnet_object.extract_features(test_input_image)
    self.assertTrue(len(feature_maps) == 1)
    print('Outputs of test_resnet_single_branch: {}'.format(feature_maps))


if __name__ == '__main__':
  tf.test.main()
