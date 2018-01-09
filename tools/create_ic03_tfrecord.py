import os
import io
import copy
import random
import re
import xml.etree.ElementTree as ET

from PIL import Image
import tensorflow as tf

from rare.utils import dataset_util
from rare.core import standard_fields as fields

flags = tf.app.flags
flags.DEFINE_string('data_dir', '', 'Root directory to raw SynthText dataset.')
flags.DEFINE_bool('ignore_difficult', True, 'Ignore words shorter than 3 or contain non-alphanumeric symbols')
flags.DEFINE_float('crop_margin', 0.2, 'Margin in percentage of word height')
FLAGS = flags.FLAGS

lexicon_size = 50
random.seed(1)


def _random_lexicon(lexicon_list, groundtruth_text, lexicon_size):
  lexicon = copy.deepcopy(lexicon_list)
  del lexicon[lexicon.index(groundtruth_text.lower())]
  random.shuffle(lexicon)
  lexicon = lexicon[:(lexicon_size-1)]
  lexicon.insert(0, groundtruth_text)
  return lexicon

def _is_difficult(word):
  assert isinstance(word, str)
  return len(word) < 3 or not re.match('^[\w]+$', word)

def create_ic03(output_path):
  writer = tf.python_io.TFRecordWriter(output_path)

  lexicon_file = os.path.join(FLAGS.data_dir, 'lexicon_full')
  with open(lexicon_file, 'r') as f:
    lexicon_list = [tline.rstrip('\n').lower() for tline in f.readlines()]

  xml_path = os.path.join(FLAGS.data_dir, 'words.xml')
  xml_root = ET.parse(xml_path).getroot()
  count = 0
  for image_node in xml_root.findall('image'):
    image_rel_path = image_node.find('imageName').text
    image_path = os.path.join(FLAGS.data_dir, image_rel_path)
    image = Image.open(image_path)
    image_w, image_h = image.size

    for i, rect in enumerate(image_node.find('taggedRectangles')):
      groundtruth_text = rect.find('tag').text.lower()
      if FLAGS.ignore_difficult and _is_difficult(groundtruth_text):
        # print('Ignoring {}'.format(groundtruth_text))
        continue

      bbox_x = float(rect.get('x'))
      bbox_y = float(rect.get('y'))
      bbox_w = float(rect.get('width'))
      bbox_h = float(rect.get('height'))
      if FLAGS.crop_margin > 0:
        margin = bbox_h * FLAGS.crop_margin
        bbox_x = bbox_x - margin
        bbox_y = bbox_y - margin
        bbox_w = bbox_w + 2 * margin
        bbox_h = bbox_h + 2 * margin
      bbox_xmin = int(round(max(0, bbox_x)))
      bbox_ymin = int(round(max(0, bbox_y)))
      bbox_xmax = int(round(min(image_w-1, bbox_x + bbox_w)))
      bbox_ymax = int(round(min(image_h-1, bbox_y + bbox_h)))

      word_crop_im = image.crop((bbox_xmin, bbox_ymin, bbox_xmax, bbox_ymax))
      im_buff = io.BytesIO()
      word_crop_im.save(im_buff, format='jpeg')
      word_crop_jpeg = im_buff.getvalue()
      crop_name = '{}:{}'.format(image_rel_path, i)

      lexicon = _random_lexicon(lexicon_list, groundtruth_text, lexicon_size)

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
        fields.TfExampleFields.lexicon: \
          dataset_util.bytes_feature(('\t'.join(lexicon)).encode('utf-8')),
      }))
      writer.write(example.SerializeToString())
      count += 1

  writer.close()
  print('{} examples created'.format(count))


if __name__ == '__main__':
  create_ic03('data/ic03_test.tfrecord')
