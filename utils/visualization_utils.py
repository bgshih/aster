import tensorflow as tf

from rare.utils import shape_utils


def tile_activation_maps_max_dimensions(maps, max_height, max_width):
  batch_size, map_height, map_width, map_depth = \
    shape_utils.combined_static_and_dynamic_shape(maps)
  num_rows = max_height // map_height
  num_cols = max_width // map_width
  return tile_activation_maps_rows_cols(maps, num_rows, num_cols)


def tile_activation_maps_rows_cols(maps, num_rows, num_cols):
  """
  Args:
    maps: [batch_size, map_height, map_width, map_depth]
  Return:
    tiled_map: [batch_size, tiled_height, tiled_width]
  """
  batch_size, map_height, map_width, map_depth = \
    shape_utils.combined_static_and_dynamic_shape(maps)

  # padding
  num_maps = num_rows * num_cols
  padded_map = tf.cond(
    tf.greater(num_maps, map_depth),
    true_fn=lambda: tf.pad(maps, [[0, 0], [0, 0], [0, 0], [0, tf.maximum(num_maps - map_depth, 0)]]),
    false_fn=lambda: maps[:,:,:,:num_maps]
  )

  # reshape to [batch_size, map_height, map_width, num_rows, num_cols]
  reshaped_map = tf.reshape(padded_map, [batch_size, map_height, map_width, num_rows, num_cols])
  
  # unstack and concat along widths
  width_concated_maps = tf.concat(
    tf.unstack(reshaped_map, axis=4), # => list of [batch_size, map_height, map_width, num_rows]
    axis=2) # => [batch_size, map_height, map_width * num_cols, num_rows]
  
  tiled_map = tf.concat(
    tf.unstack(width_concated_maps, axis=3), # => list of [batch_size, map_height, map_width * num_cols]
    axis=1) # => [batch_size, map_height * num_rows, map_width * num_cols]

  tiled_map = tf.expand_dims(tiled_map, axis=3)

  return tiled_map
