import logging
import sys
import os
import random
import warnings
from PIL import Image, ImageDraw
import io

import tensorflow as tf
import numpy as np
import scipy.io as sio
from tqdm import tqdm

from rare.utils import dataset_util
from rare.core import standard_fields as fields

margin_ratio = 0.1
num_keypoints = 128

flags = tf.app.flags
flags.DEFINE_string('data_dir', '', 'Root directory to raw SynthText dataset.')
flags.DEFINE_integer('start_index', 0, 'Start image index.')
flags.DEFINE_integer('num_images', -1, 'Number of images to create. Default is all remaining.')
flags.DEFINE_integer('shuffle', 0, 'Shuffle images.')
flags.DEFINE_string('output_path', '', 'Path to output TFRecord.')
flags.DEFINE_integer('num_dump_images', 0, 'Number of images to dump for debugging')
FLAGS = flags.FLAGS


class SynthTextCreator(object):
  
  def __init__(self, output_path, data_dir, start_index,
               num_images=0, shuffle=False, num_dump_images=0):
    self._data_dir = data_dir
    self._start_index = start_index
    self._num_images = num_images
    self._shuffle = shuffle
    self._num_dump_images = num_dump_images

    self._writer = tf.python_io.TFRecordWriter(output_path)

    self._sample_count = 0
    self._skipped_sample_count = 0

  def create():
    # load groundtruth file
    groundtruth_path = os.path.join(self._data_dir, 'gt.mat')
    if not os.path.exists(groundtruth_path):
      raise ValueError('Could not find groundtruth file: {}'.format(groundtruth_path))
    print('Loading groundtruth...')
    groundtruth = sio.loadmat(groundtruth_path)

    num_images = groundtruth['wordBB'].shape[1] - self._start_index
    if self._num_images > 0:
      num_images = min(num_images, self._num_images)
    
    indices = list(range(FLAGS.start_index, FLAGS.start_index + num_images))
    if self._shuffle:
      random.shuffle(indices)
    
    for index in tqdm(indices):
      image_rel_path = str(groundtruth['imnames'][0, index][0])

      text = groundtruth['txt'][0, index]
      words = []
      for text_line in text:
        text_line = str(text_line)
        line_words = ('\n'.join(text_line.split())).split('\n')
        words.extend(line_words)
      if len(words) != num_bboxes:
        raise ValueError('Number of words and bboxes mismtach: {} vs {}'.format(len(words), num_bboxes))



      word_polygons = groundtruth['wordBB'][0, index]
      if word_polygons.ndim == 2:
        word_polygons = np.expand_dims(word_polygons, axis=2)
      word_polygons = np.transpose(word_polygons, axes=[2,1,0])
      char_polygons = np.transpose(groundtruth['charBB'][0, index], axes=[2,1,0])
      self._create_samples_of_an_image(image_rel_path, text, word_polygons, char_polygons)

  def _create_samples_of_an_image(image_rel_path, words, word_polygons, char_polygons):
    """
    Args:
      image_rel_path: Python string. Image relative path.
      words: string array of shape [num_words]. Groundtruth words.
      word_polygons: array of shape [num_words, ]. Groundtruth word polygons
      
    """


    image_path = os.path.join(self._data_dir, image_rel_path)
    im = Image.open(image_path)
    im_width, im_height = im.size

    bbox_xymin = np.min(word_polygons, axis=1)
    bbox_xymax = np.max(word_polygons, axis=1)
    bbox_wh = bbox_xymax - bbox_xymin
    bbox_margin = np.expand_dims(
      margin_ratio * np.sqrt(bbox_wh[:,0] * bbox_wh[:,1]),
      axis=1)
    enlarged_bbox_xymin = np.maximum(
      bbox_xymin - bbox_margin,
      np.asarray([[0, 0]], dtype=np.float32))
    enlarged_bbox_xymax = np.minimum(
      bbox_xymax + bbox_margin,
      np.asarray([[im_width-1, im_height-1]], dtype=np.float32))
    bbox_array = np.concatenate([enlarged_bbox_xymin, enlarged_bbox_xymax], axis=1)
    bbox_array = np.round(bbox_array)












def main(_):
  writer = tf.python_io.TFRecordWriter(FLAGS.output_path)

  # load groundtruth file
  groundtruth_path = os.path.join(FLAGS.data_dir, 'gt.mat')
  if not os.path.exists(groundtruth_path):
    raise ValueError('Could not find groundtruth file: {}'.format(groundtruth_path))
  print('Loading groundtruth...')
  groundtruth = sio.loadmat(groundtruth_path)
  
  num_images = groundtruth['wordBB'].shape[1] - FLAGS.start_index

  if FLAGS.num_images > 0:
    num_images = min(num_images, FLAGS.num_images)

  indices = list(range(FLAGS.start_index, FLAGS.start_index + num_images))
  if FLAGS.shuffle:
    random.shuffle(indices)

  count = 0
  skipped = 0
  dump_images_count = 0

  for index in tqdm(indices):
    image_rel_path = str(groundtruth['imnames'][0, index][0])
    text = groundtruth['txt'][0, index]
    word_polygons = groundtruth['wordBB'][0, index]
    char_polygons = np.transpose(groundtruth['charBB'][0, index], axes=[2,1,0])
    _create_samples_of_an_image(writer, image_rel_path, text, word_polygons, char_polygons)

  for index in tqdm(indices):
    # try:
    image_rel_path = str(groundtruth['imnames'][0, index][0])
    image_path = os.path.join(FLAGS.data_dir, image_rel_path)

    # load image jpeg data
    im = Image.open(image_path)
    im_width, im_height = im.size
    
    # word polygons
    word_polygons = groundtruth['wordBB'][0, index]
    if word_polygons.ndim == 2:
      word_polygons = np.expand_dims(word_polygons, axis=2)
    word_polygons = np.transpose(word_polygons, axes=[2,1,0])
    bbox_xymin = np.min(word_polygons, axis=1)
    bbox_xymax = np.max(word_polygons, axis=1)
    bbox_wh = bbox_xymax - bbox_xymin
    bbox_margin = np.expand_dims(
      margin_ratio * np.sqrt(bbox_wh[:,0] * bbox_wh[:,1]),
      axis=1)
    enlarged_bbox_xymin = np.maximum(
      bbox_xymin - bbox_margin,
      np.asarray([[0, 0]], dtype=np.float32))
    enlarged_bbox_xymax = np.minimum(
      bbox_xymax + bbox_margin,
      np.asarray([[im_width-1, im_height-1]], dtype=np.float32))
    bbox_array = np.concatenate([enlarged_bbox_xymin, enlarged_bbox_xymax], axis=1)
    bbox_array = np.round(bbox_array)
    num_bboxes = bbox_array.shape[0]

    # words
    text = groundtruth['txt'][0, index]
    words = []
    for text_line in text:
      text_line = str(text_line)
      line_words = ('\n'.join(text_line.split())).split('\n')
      words.extend(line_words)
    if len(words) != num_bboxes:
      raise ValueError('Number of words and bboxes mismtach: {} vs {}'.format(len(words), num_bboxes))

    # char polygons for every word
    all_char_polygons = np.transpose(groundtruth['charBB'][0, index], axes=[2,1,0])
    char_polygons_list = []
    offset = 0
    for word in words:
      word_len = len(word)
      char_polygons_list.append(all_char_polygons[offset:offset+word_len])
      offset += word_len
    if offset != all_char_polygons.shape[0]:
      raise ValueError('Offset mismtach: {} vs {}'.format(offset, all_char_polygons.shape[0]))

    def _fit_and_divide(points):
      # points: [num_points, 2]
      degree = 2 if points.shape[0] > 2 else 1
      coeffs = np.polyfit(points[:,0], points[:,1], degree)
      poly_fn = np.poly1d(coeffs)
      xmin, xmax = np.min(points[:,0]), np.max(points[:,0])
      xs = np.linspace(xmin, xmax, num=(num_keypoints // 2))
      ys = poly_fn(xs)
      return np.stack([xs, ys], axis=1)

    image_w, image_h = im.size

    for i, bbox in enumerate(bbox_array):
      try:
        # crop image and encode to jpeg
        crop_coordinates = tuple(bbox.astype(np.int))
        crop_xmin, crop_ymin, crop_xmax, crop_ymax = crop_coordinates
        crop_w, crop_h = crop_xmax - crop_xmin, crop_ymax - crop_ymin
        if (crop_xmin < 0 or
            crop_ymin < 0 or
            crop_xmax >= image_w or
            crop_ymax >= image_h or
            crop_w <= 0 or
            crop_h <= 0):
          raise ValueError('Invalid crop box {}'.format(crop_coordinates))
        crop_w = crop_xmax - crop_xmin
        crop_h = crop_ymax - crop_ymin
        if crop_w * crop_h < 20:
          raise ValueError('Crop area too small: {}x{}'.format(crop_w, crop_h))
        
        word_crop_im = im.crop(crop_coordinates)
        im_buff = io.BytesIO()
        word_crop_im.save(im_buff, format='jpeg')
        word_crop_jpeg = im_buff.getvalue()
        crop_name = '{}:{}'.format(image_rel_path, i)
        word_crop_w, word_crop_h = word_crop_im.size

        # fit curves to chars polygon points and divide the curve
        char_polygons = char_polygons_list[i]
        crop_xymin = [crop_xmin, crop_ymin]
        rel_char_polygons = char_polygons - [[crop_xymin]]
        with warnings.catch_warnings():
          warnings.simplefilter('error', np.RankWarning)
          try:
            top_curve_points = _fit_and_divide(rel_char_polygons[:,:2,:].reshape([-1, 2]))
            bottom_curve_points = _fit_and_divide(rel_char_polygons[:,2:,:].reshape([-1, 2]))
          except np.RankWarning:
            raise ValueError('Bad polyfit.')

        curve_points = np.concatenate([top_curve_points, bottom_curve_points], axis=0)
        flat_curve_points = curve_points.flatten().tolist()

        if FLAGS.num_dump_images > 0 and dump_images_count < FLAGS.num_dump_images:
          def _draw_cross(draw, center, size=2):
            left_pt = tuple(center - [size, 0])
            right_pt = tuple(center + [size, 0])
            top_pt = tuple(center - [0, size])
            bottom_pt = tuple(center + [0, size])
            draw.line([top_pt, bottom_pt], width=1, fill='#ffffff')
            draw.line([left_pt, right_pt], width=1, fill='#ffffff')
          save_fname = 'rare/vis/{}_{}.jpg'.format(count, words[i])
          draw = ImageDraw.Draw(word_crop_im)
          for pts in curve_points:
            _draw_cross(draw, pts)
          word_crop_im.save(save_fname)
          dump_images_count += 1

        # write an example
        example = tf.train.Example(features=tf.train.Features(feature={
            fields.TfExampleFields.image_encoded: \
                dataset_util.bytes_feature(word_crop_jpeg),
            fields.TfExampleFields.image_format: \
                dataset_util.bytes_feature('jpeg'.encode('utf-8')),
            fields.TfExampleFields.filename: \
                dataset_util.bytes_feature(crop_name.encode('utf-8')),
            fields.TfExampleFields.channels: \
                dataset_util.int64_feature(3),
            fields.TfExampleFields.colorspace: \
                dataset_util.bytes_feature('rgb'.encode('utf-8')),
            fields.TfExampleFields.width: \
                dataset_util.int64_feature(word_crop_w),
            fields.TfExampleFields.height: \
                dataset_util.int64_feature(word_crop_h),
            fields.TfExampleFields.transcript: \
                dataset_util.bytes_feature(words[i].encode('utf-8')),
            fields.TfExampleFields.keypoints: \
                dataset_util.float_list_feature(flat_curve_points),
        }))

        writer.write(example.SerializeToString())
        count += 1
      except ValueError as err:
        print("ValueError: {}".format(err))
        skipped += 1
        continue
        # except:
        #   print("Unexpected error:", sys.exc_info()[0])
        #   skipped += 1
        #   continue

    # except:
    #   print("Image #{} error:".format(index), sys.exc_info()[0])
    #   continue
  
  print('{} samples created, {} skipped'.format(count, skipped))
  writer.close()


if __name__ == '__main__':
  tf.app.run()
