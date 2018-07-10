import os
import io
import xml.etree.ElementTree as ET

from PIL import Image
import tensorflow as tf

from aster.utils import dataset_util
from aster.core import standard_fields as fields

flags = tf.app.flags
flags.DEFINE_string('data_dir', '', 'Root directory to raw SynthText dataset.')
FLAGS = flags.FLAGS

def create_svt_perspective(output_path):
  writer = tf.python_io.TFRecordWriter(output_path)
  image_list_file = os.path.join(FLAGS.data_dir, 'imagelist.txt')
  with open(image_list_file, 'r') as f:
    tlines = [tline.rstrip('\n') for tline in f.readlines()]

  count = 0

  for tline in tlines:
    image_rel_path, groundtruth_text, lexicon_length, lexicon = \
      tline.split(' ')
    groundtruth_text = groundtruth_text.lower()
    lexicon_length = int(lexicon_length)
    lexicon_list = [w.lower() for w in lexicon.split(',')]

    image_path = os.path.join(FLAGS.data_dir, image_rel_path)
    with open(image_path, 'rb') as f:
      image_jpeg = f.read()

    example = tf.train.Example(features=tf.train.Features(feature={
        fields.TfExampleFields.image_encoded: \
          dataset_util.bytes_feature(image_jpeg),
        fields.TfExampleFields.image_format: \
          dataset_util.bytes_feature('jpeg'.encode('utf-8')),
        fields.TfExampleFields.filename: \
          dataset_util.bytes_feature(image_rel_path.encode('utf-8')),
        fields.TfExampleFields.channels: \
          dataset_util.int64_feature(3),
        fields.TfExampleFields.colorspace: \
          dataset_util.bytes_feature('rgb'.encode('utf-8')),
        fields.TfExampleFields.transcript: \
          dataset_util.bytes_feature(groundtruth_text.encode('utf-8')),
        fields.TfExampleFields.lexicon: \
          dataset_util.bytes_feature(('\t'.join(lexicon_list)).encode('utf-8')),
      }))
    writer.write(example.SerializeToString())
    count += 1

  writer.close()
  print('{} examples created'.format(count))

if __name__ == '__main__':
  create_svt_perspective('data/svt_perspective_test.tfrecord')
