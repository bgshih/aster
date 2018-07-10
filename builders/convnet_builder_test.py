import tensorflow as tf
from google.protobuf import text_format

from aster.builders import convnet_builder
from aster.protos import convnet_pb2
from aster.convnets import crnn_net
from aster.convnets import resnet
from aster.convnets import stn_convnet

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

  def test_build_stn_convnet(self):
    text_proto = """
    stn_convnet {
      conv_hyperparams {
        op: CONV
        regularizer { l2_regularizer { weight: 1e-4 } }
        initializer { variance_scaling_initializer { } }
        batch_norm { decay: 0.99 }
      }
    }
    """
    convnet_proto = convnet_pb2.Convnet()
    text_format.Merge(text_proto, convnet_proto)
    convnet_object = convnet_builder.build(convnet_proto, True)
    self.assertTrue(
      isinstance(convnet_object, stn_convnet.StnConvnet))
    test_image_shape = [2, 64, 128, 3]
    test_input_image = tf.random_uniform(
      test_image_shape,
      minval=0,
      maxval=255.0,
      dtype=tf.float32,
      seed=1
    )
    feature_maps = convnet_object.extract_features(test_input_image)
    self.assertTrue(len(feature_maps) == 1)
    print('Outputs of test_build_stn_convnet: {}'.format(feature_maps))

  def test_build_stn_convnet_tiny(self):
    text_proto = """
    stn_convnet {
      conv_hyperparams {
        op: CONV
        regularizer { l2_regularizer { weight: 1e-4 } }
        initializer { variance_scaling_initializer { } }
        batch_norm { decay: 0.99 }
      }
      tiny: true
    }
    """
    convnet_proto = convnet_pb2.Convnet()
    text_format.Merge(text_proto, convnet_proto)
    convnet_object = convnet_builder.build(convnet_proto, True)
    self.assertTrue(
      isinstance(convnet_object, stn_convnet.StnConvnetTiny))
    test_image_shape = [2, 64, 128, 3]
    test_input_image = tf.random_uniform(
      test_image_shape,
      minval=0,
      maxval=255.0,
      dtype=tf.float32,
      seed=1
    )
    feature_maps = convnet_object.extract_features(test_input_image)
    self.assertTrue(len(feature_maps) == 1)
    print('Outputs of test_build_stn_convnet_tiny: {}'.format(feature_maps))


if __name__ == '__main__':
  tf.test.main()
