import tensorflow as tf

from aster.utils import visualization_utils

class VisualizationUtilsTest(tf.test.TestCase):

  def test_tile_activation_maps_with_padding(self):
    test_maps = tf.random_uniform([64, 32, 100, 16])
    tiled_map = visualization_utils.tile_activation_maps_rows_cols(test_maps, 5, 5)

    with self.test_session() as sess:
      tiled_map_output = tiled_map.eval()
      self.assertAllEqual(tiled_map_output.shape, [64, 32 * 5, 100 * 5, 1])
  
  def test_tile_activation_maps_with_slicing(self):
    test_maps = tf.random_uniform([64, 32, 100, 16])
    tiled_map = visualization_utils.tile_activation_maps_rows_cols(test_maps, 5, 1)

    with self.test_session() as sess:
      tiled_map_output = tiled_map.eval()
      self.assertAllEqual(tiled_map_output.shape, [64, 32 * 5, 100 * 1, 1])
  
  def test_tile_activation_maps_max_sizes(self):
    test_maps = tf.random_uniform([64, 32, 100, 16])
    tiled_map = visualization_utils.tile_activation_maps_max_dimensions(
      test_maps, 512, 512)
    
    with self.test_session() as sess:
      tiled_map_output = tiled_map.eval()
      self.assertAllEqual(tiled_map_output.shape, [64, 512, 500, 1])


if __name__ == '__main__':
  tf.test.main()
