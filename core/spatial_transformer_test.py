import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

from rare.core import spatial_transformer


class SpatialTransformerTest(tf.test.TestCase):
  
  def test_batch_transform(self):
    transformer = spatial_transformer.SpatialTransformer(
      output_image_size=(32, 100),
      num_control_points=6,
      init_bias_pattern='identity',
      margin=0.05
    )
    test_input_ctrl_pts = np.array([
      [
        [0.1, 0.4], [0.5, 0.1], [0.9, 0.4],
        [0.1, 0.9], [0.5, 0.6], [0.9, 0.9]
      ],
      [
        [0.1, 0.1], [0.5, 0.4], [0.9, 0.1],
        [0.1, 0.6], [0.5, 0.9], [0.9, 0.6]
      ],
      [
        [0.1, 0.1], [0.5, 0.1], [0.9, 0.1],
        [0.1, 0.9], [0.5, 0.9], [0.9, 0.9],
      ]
    ], dtype=np.float32)
    test_im = Image.open('rare/data/test_image.jpg').resize((128, 128))
    test_image_array = np.array(test_im)
    test_image_array = np.array([test_image_array, test_image_array, test_image_array])
    test_images = tf.cast(tf.constant(test_image_array), tf.float32)
    test_images = (test_images / 128.0) - 1.0

    sampling_grid = transformer._batch_generate_grid(test_input_ctrl_pts)
    rectified_images = transformer._batch_sample(test_images, sampling_grid)

    output_ctrl_pts = transformer._output_ctrl_pts
    with self.test_session() as sess:
      outputs = sess.run({
        'sampling_grid': sampling_grid,
        'rectified_images': rectified_images
      })
    
    rectified_images_ = (outputs['rectified_images'] + 1.0) * 128.0

    if True:
      plt.figure()
      plt.subplot(3,4,1)
      plt.scatter(test_input_ctrl_pts[0,:,0], test_input_ctrl_pts[0,:,1])
      plt.subplot(3,4,2)
      plt.scatter(output_ctrl_pts[:,0], output_ctrl_pts[:,1])
      plt.subplot(3,4,3)
      plt.scatter(outputs['sampling_grid'][0,:,0], outputs['sampling_grid'][0,:,1], marker='+')
      plt.subplot(3,4,4)
      plt.imshow(rectified_images_[0].astype(np.uint8))

      plt.subplot(3,4,5)
      plt.scatter(test_input_ctrl_pts[1,:,0], test_input_ctrl_pts[1,:,1])
      plt.subplot(3,4,6)
      plt.scatter(output_ctrl_pts[:,0], output_ctrl_pts[:,1])
      plt.subplot(3,4,7)
      plt.scatter(outputs['sampling_grid'][1,:,0], outputs['sampling_grid'][1,:,1], marker='+')
      plt.subplot(3,4,8)
      plt.imshow(rectified_images_[1].astype(np.uint8))

      plt.subplot(3,4,9)
      plt.scatter(test_input_ctrl_pts[2,:,0], test_input_ctrl_pts[2,:,1])
      plt.subplot(3,4,10)
      plt.scatter(output_ctrl_pts[:,0], output_ctrl_pts[:,1])
      plt.subplot(3,4,11)
      plt.scatter(outputs['sampling_grid'][2,:,0], outputs['sampling_grid'][2,:,1], marker='+')
      plt.subplot(3,4,12)
      plt.imshow(rectified_images_[2].astype(np.uint8))

      plt.show()


if __name__ == '__main__':
  tf.test.main()
