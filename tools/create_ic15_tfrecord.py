# -*- coding: utf-8 -*-

import os
import io
import random
import re
import glob

from PIL import Image
import tensorflow as tf

from aster.utils import dataset_util
from aster.core import standard_fields as fields

flags = tf.app.flags
flags.DEFINE_string('data_dir', '/home/mkyang/dataset/recognition/icdar2015/', 'Root directory to raw SynthText dataset.')
flags.DEFINE_bool('exclude_difficult', False, 'Excluding non-alphanumeric examples.')
flags.DEFINE_string('output_path', 'data/ic15_test_all.tfrecord', 'Output tfrecord path.')
FLAGS = flags.FLAGS

def _is_difficult(word):
  assert isinstance(word, str)
  return not re.match('^[\w]+$', word)

def char_check(word):
  if not word.isalnum():
    return False
  else:
    for char in word:
      if char < ' ' or char > '~':
        return False
  return True

def create_ic15(output_path):
  writer = tf.python_io.TFRecordWriter(output_path)

  groundtruth_file_path = os.path.join(FLAGS.data_dir, 'test_groundtruth_all.txt')
  
  count = 0
  with open(groundtruth_file_path, 'r') as f:
    lines = f.readlines()
    img_gts = [line.strip() for line in lines]
    for img_gt in img_gts:
      img_rel_path, gt = img_gt.split(' ', 1)
      if FLAGS.exclude_difficult and not char_check(gt):
        continue
      img_path = os.path.join(FLAGS.data_dir, img_rel_path)
      img = Image.open(img_path)
      img_buff = io.BytesIO()
      img.save(img_buff, format='jpeg')
      word_crop_jpeg = img_buff.getvalue()
      crop_name = os.path.basename(img_path)

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
        fields.TfExampleFields.transcript: \
          dataset_util.bytes_feature(gt.encode('utf-8')),
      }))
      writer.write(example.SerializeToString())
      count += 1
  
  writer.close()
  print('{} examples created'.format(count))

if __name__ == '__main__':
  create_ic15(FLAGS.output_path)
