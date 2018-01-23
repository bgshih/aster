import os
import io
import random
import re
import glob

from PIL import Image
import tensorflow as tf

from rare.utils import dataset_util
from rare.core import standard_fields as fields

flags = tf.app.flags
flags.DEFINE_string('data_dir', '/home/mkyang/dataset/recognition/icdar2015/', 'Root directory to raw SynthText dataset.')
flags.DEFINE_float('crop_margin', 0.15, 'Margin in percentage of word height')
FLAGS = flags.FLAGS

def _is_difficult(word):
  assert isinstance(word, str)
  return not re.match('^[\w]+$', word)

def create_ic15(output_path):
  writer = tf.python_io.TFRecordWriter(output_path)

  groundtruth_file_path = os.path.join(FLAGS.data_dir, 'test_groundtruth.txt')
  
  count = 0
  with open(groundtruth_file_path, 'r') as f:
    img_gts = f.readlines()
    img_gts = [img_gt.strip() for img_gt in img_gts]
    for img_gt in img_gts:
      img_rel_path, gt = img_gt.split(' ', 1)
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
  create_ic15('rare/data/ic15_test.tfrecord')
