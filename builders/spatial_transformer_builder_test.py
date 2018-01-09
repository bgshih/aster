import tensorflow as tf

from google.protobuf import text_format

from rare.protos import spatial_transformer_pb2
from rare.builders import spatial_transformer_builder


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
    """
    config = spatial_transformer_pb2.SpatialTransformer()
    text_format.Merge(text_proto, config)
    spatial_transformer_object = spatial_transformer_builder.build(config, True)

    test_input_images = tf.random_uniform(
      [2, 64, 512, 3], minval=0, maxval=255, dtype=tf.float32)
    rectified_images = spatial_transformer_object.batch_transform(test_input_images)

    with self.test_session() as sess:
      sess.run(tf.global_variables_initializer())
      sess_outputs = sess.run({
        'rectified_images': rectified_images
      })
      self.assertEqual(sess_outputs['rectified_images'].shape, (2, 32, 100, 3))


if __name__ == '__main__':
  tf.test.main()
