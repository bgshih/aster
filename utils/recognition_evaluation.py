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

    correct_mask = (
      np.char.lower(all_recognition_text_array) ==
      np.char.lower(all_groundtruth_text_array))

    # print incorrect predictions
    incorrect_recognition_text_array = all_recognition_text_array[np.logical_not(correct_mask)]
    incorrect_groundtruth_text_array = all_groundtruth_text_array[np.logical_not(correct_mask)]

    n_print = min(20, incorrect_recognition_text_array.shape[0])
    print('*** Groundtruth => Prediction ***')
    for i in range(n_print):
      print('{} => {}'.format(
        incorrect_groundtruth_text_array[i].decode('utf-8'),
        incorrect_recognition_text_array[i].decode('utf-8')))
    print('**********************************')

    # case insensitive accuracy
    case_insensitive_accuracy = np.count_nonzero(correct_mask) / num_samples
    return case_insensitive_accuracy
