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

  def add_single_image_recognition_info(self, image_key,
                                        recognition_text,
                                        groundtruth_text):
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

    self.all_recognition_text.append(recognition_text)
    self.all_groundtruth_text.append(groundtruth_text)

  def evaluate_all(self):
    num_samples = len(self.all_recognition_text)
    all_recognition_text_array = np.asarray(self.all_recognition_text)
    all_groundtruth_text_array = np.asarray(self.all_groundtruth_text)

    all_groundtruth_text_array = np.char.lower(all_groundtruth_text_array)

    case_insensitive_accuracy = np.count_nonzero(
      all_recognition_text_array == all_groundtruth_text_array) / num_samples
    return case_insensitive_accuracy
