import os
import string

import tensorflow as tf

from rare.core import label_map
from rare.protos import label_map_pb2


def build(config):
  if not isinstance(config, label_map_pb2.LabelMap):
    raise ValueError('config not of type label_map_pb2.LabelMap')

  character_set = _build_character_set(config.character_set)
  label_map_object = label_map.LabelMap(character_set=character_set)
  return label_map_object


def _build_character_set(config):
  if not isinstance(config, label_map_pb2.CharacterSet):
    raise ValueError('config not of type label_map_pb2.CharacterSet')

  source_oneof = config.WhichOneof('source_oneof')
  character_set_string = None
  if source_oneof == 'text_file':
    file_path = config.text_file
    with open(file_path, 'r') as f:
      character_set_string = f.read()
  elif source_oneof == 'text_string':
    character_set_string = config.text_string

  if not config.delimiter:
    character_set = list(character_set_string)
  else:
    character_set = character_set_string.split()
  return character_set
