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
flags.DEFINE_string('data_dir', '', 'Root directory to raw SynthText dataset.')
flags.DEFINE_float('crop_margin', 0.15, 'Margin in percentage of word height')
FLAGS = flags.FLAGS

def _is_difficult(word):
  assert isinstance(word, str)
  return not re.match('^[\w]+$', word)

def create_ic13(output_path):
  writer = tf.python_io.TFRecordWriter(output_path)

  groundtruth_dir = os.path.join(FLAGS.data_dir, 'Challenge2_Test_Task1_GT')
  groundtruth_files = glob.glob(os.path.join(groundtruth_dir, '*.txt'))
  
  count = 0
  for groundtruth_file in groundtruth_files:
    image_id = re.match(r'.*gt_img_(\d+).txt$', groundtruth_file).group(1)
    image_rel_path = 'img_{}.jpg'.format(image_id)
    image_path = os.path.join(FLAGS.data_dir, 'Challenge2_Test_Task12_Images', image_rel_path)
    image = Image.open(image_path)
    image_w, image_h = image.size

    with open(groundtruth_file, 'r') as f:
      groundtruth = f.read()

    matches = re.finditer(r'^(\d+),\s(\d+),\s(\d+),\s(\d+),\s\"(.+)\"$', groundtruth, re.MULTILINE)
    for i, match in enumerate(matches):
      bbox_xmin = float(match.group(1))
      bbox_ymin = float(match.group(2))
      bbox_xmax = float(match.group(3))
      bbox_ymax = float(match.group(4))
      groundtruth_text = match.group(5)

      if _is_difficult(groundtruth_text):
        continue

      if FLAGS.crop_margin > 0:
        bbox_h = bbox_ymax - bbox_ymin
        margin = bbox_h * FLAGS.crop_margin
        bbox_xmin = bbox_xmin - margin
        bbox_ymin = bbox_ymin - margin
        bbox_xmax = bbox_xmax + margin
        bbox_ymax = bbox_ymax + margin
      bbox_xmin = int(round(max(0, bbox_xmin)))
      bbox_ymin = int(round(max(0, bbox_ymin)))
      bbox_xmax = int(round(min(image_w-1, bbox_xmax)))
      bbox_ymax = int(round(min(image_h-1, bbox_ymax)))

      word_crop_im = image.crop((bbox_xmin, bbox_ymin, bbox_xmax, bbox_ymax))
      im_buff = io.BytesIO()
      word_crop_im.save(im_buff, format='jpeg')
      word_crop_jpeg = im_buff.getvalue()
      crop_name = '{}:{}'.format(image_rel_path, i)

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
          dataset_util.bytes_feature(groundtruth_text.encode('utf-8')),
      }))
      writer.write(example.SerializeToString())
      count += 1
  
  writer.close()
  print('{} examples created'.format(count))

if __name__ == '__main__':
  create_ic13('data/ic13_test.tfrecord')
