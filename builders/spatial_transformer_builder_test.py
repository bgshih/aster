import numpy as np
import tensorflow as tf

from google.protobuf import text_format

from aster.protos import spatial_transformer_pb2
from aster.builders import spatial_transformer_builder


class SpatialTransformerBuilderTest(tf.test.TestCase):

  def test_build_spatial_transformer(self):
    text_proto = """
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
    init_bias_pattern: "slope"
    summarize_activations: true
    """
    config = spatial_transformer_pb2.SpatialTransformer()
    text_format.Merge(text_proto, config)
    spatial_transformer_object = spatial_transformer_builder.build(config, True)
    self.assertTrue(spatial_transformer_object._summarize_activations == True)

    test_input_images = tf.random_uniform(
      [2, 64, 512, 3], minval=0, maxval=255, dtype=tf.float32)
    output_dict = spatial_transformer_object.batch_transform(test_input_images)

    with self.test_session() as sess:
      sess.run(tf.global_variables_initializer())
      sess_outputs = sess.run({
        'rectified_images': output_dict['rectified_images'],
        'control_points': output_dict['control_points'],
      })
      self.assertEqual(sess_outputs['rectified_images'].shape, (2, 32, 100, 3))

      init_bias = spatial_transformer_object._init_bias
      init_ctrl_pts = (1. / (1. + np.exp(-init_bias))).reshape(20, 2)
      self.assertAllClose(sess_outputs['control_points'][0], init_ctrl_pts)

if __name__ == '__main__':
  tf.test.main()
