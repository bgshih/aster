import tensorflow as tf
import numpy as np

from google.protobuf import text_format
from rare.core import loss
from rare.builders import loss_builder
from rare.protos import loss_pb2


class LossTest(tf.test.TestCase):

  def test_build_seq_loss(self):
    loss_text_proto = """
      sequence_cross_entropy_loss {
        sequence_normalize: false
        sample_normalize: true
        weight: 0.5
      }
    """
    loss_proto = loss_pb2.Loss()
    text_format.Merge(loss_text_proto, loss_proto)
    loss_object = loss_builder.build(loss_proto)

    test_logits = tf.constant(
      [
        [
          [0.0, 1.0],
          [0.5, 0.5],
          [0.3, 0.7],
        ],
        [
          [0.0, -1.0],
          [1.0, 10.0],
          [1.0, 20.0],
        ],
      ],
      dtype=tf.float32
    )
    test_labels = tf.constant(
      [
        [0, 1, 0],
        [0, 0, 0]
      ],
      dtype=tf.int32
    )
    test_lengths = tf.constant(
      [3, 1],
      dtype=tf.int32
    )
    loss_tensor = loss_object(test_logits, test_labels, test_lengths, scope='loss')

    with self.test_session() as sess:
      outputs = sess.run({
        'loss': loss_tensor
      })
      print(outputs)

  def test_build_reg_loss(self):
    loss_text_proto = """
      l2_regression_loss {
        weight: 1.0
      }
    """
    loss_proto = loss_pb2.Loss()
    text_format.Merge(loss_text_proto, loss_proto)
    loss_object = loss_builder.build(loss_proto)
    self.assertTrue(isinstance(loss_object, loss.L2RegressionLoss))

    prediction = tf.constant(np.random.uniform(0, 1, (2, 20)))
    target = tf.constant(np.random.uniform(0, 1, (2, 20)))
    loss_tensor = loss_object(prediction, target)
    with self.test_session() as sess:
      print({'loss': loss_tensor.eval()})

if __name__ == '__main__':
  tf.test.main()
