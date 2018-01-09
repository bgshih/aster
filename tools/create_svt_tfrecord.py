import os
import io
import xml.etree.ElementTree as ET

from PIL import Image
import tensorflow as tf

from rare.utils import dataset_util
from rare.core import standard_fields as fields

flags = tf.app.flags
flags.DEFINE_string('data_dir', '', 'Root directory to raw SynthText dataset.')
flags.DEFINE_float('crop_margin', 0.05, 'Margin in percent of word height')
FLAGS = flags.FLAGS


def create_svt_subset(output_path):
  writer = tf.python_io.TFRecordWriter(output_path)
  test_xml_path = os.path.join(FLAGS.data_dir, 'test.xml')
  count = 0
  xml_root = ET.parse(test_xml_path).getroot()
  for image_node in xml_root.findall('image'):
    image_rel_path = image_node.find('imageName').text
    lexicon = image_node.find('lex').text.lower()
    lexicon = lexicon.split(',')
    image_path = os.path.join(FLAGS.data_dir, image_rel_path)
    image = Image.open(image_path)
    image_w, image_h = image.size

    for i, rect in enumerate(image_node.find('taggedRectangles')):
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

      groundtruth_text = rect.find('tag').text.lower()
    
      example = tf.train.Example(features=tf.train.Features(feature={
        fields.TfExampleFields.image_encoded: \
          dataset_util.bytes_feature(word_crop_jpeg),
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
          dataset_util.bytes_feature(('\t'.join(lexicon)).encode('utf-8')),
      }))
      writer.write(example.SerializeToString())
      count += 1

  writer.close()
  print('{} examples created'.format(count))

if __name__ == '__main__':
  create_svt_subset('data/svt_test.tfrecord')
