class InputDataFields(object):
  image = 'image'
  original_image = 'original_image'
  key = 'key'
  source_id = 'source_id'
  filename = 'filename'
  groundtruth_transcript = 'groundtruth_transcript'


class TfExampleFields(object):
  image_encoded = 'image/encoded'
  image_format = 'image/format'  # format is reserved keyword
  filename = 'image/filename'
  channels = 'image/channels'
  colorspace = 'image/colorspace'
  height = 'image/height'
  width = 'image/width'
  source_id = 'image/source_id'
  transcript = 'image/transcript'
