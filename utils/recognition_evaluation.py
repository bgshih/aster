import string
import logging

import numpy as np

class RecognitionEvaluation(object):
  def __init__(self):
    self.image_keys = set()
    self.all_recognition_text = []
    self.all_groundtruth_text = []

  def clear(self):
    self.image_keys = set()
    self.all_recognition_text = []
    self.all_groundtruth_text = []

  def add_single_image_recognition_info(self, image_key, recognition_text, groundtruth_text):
    """
    Args:
      image_key: Python string
      recognition_text: Numpy scalar of string type
      groundtruth_text: Numpy scalar of string type
    """
    if image_key in self.image_keys:
      logging.warning('{} already evaluated'.format(image_key))
      return
    self.image_keys.add(image_key)

    self.all_recognition_text.append(recognition_text.decode('utf-8'))
    self.all_groundtruth_text.append(groundtruth_text.decode('utf-8'))

  def evaluate_all(self):
    num_samples = len(self.all_recognition_text)

    def _normalize_text(text):
      text = ''.join(filter(lambda x: x in (string.digits + string.ascii_letters), text))
      return text.lower()

    num_correct = 0
    num_incorrect = 0
    incorrect_pairs = []
    for i in range(num_samples):
      recogition = _normalize_text(self.all_recognition_text[i])
      groundtruth = _normalize_text(self.all_groundtruth_text[i])
      if recogition == groundtruth:
        num_correct += 1
      else:
        num_incorrect += 1
        incorrect_pairs.append((recogition, groundtruth))
    num_print = min(len(incorrect_pairs), 100)
    print('*** Groundtruth => Prediction ***')
    for i in range(num_print):
      recogition, groundtruth = incorrect_pairs[i]
      print('{} => {}'.format(groundtruth, recogition))
    print('**********************************')
    case_insensitive_accuracy = num_correct / (num_correct + num_incorrect)
    return case_insensitive_accuracy
