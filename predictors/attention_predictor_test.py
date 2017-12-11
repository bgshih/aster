import string

import tensorflow as tf

from rare.predictors import attention_predictor
from rare.core import label_map

class AttentionPredictorTest(tf.test.TestCase):

  def test_attention_predictor(self):
    rnn_cell_object = tf.contrib.rnn.GRUCell(256)
    label_map_object = label_map.LabelMap(character_set=list(string.ascii_lowercase))

    predictor_object = attention_predictor.BahdanauAttentionPredictor(
      rnn_cell_object,
      num_attention_units=256,
      max_num_steps=50,
      is_training=True)

    test_feature_map = tf.random_uniform([2, 1, 15, 32], minval=0, maxval=3, seed=2)
    test_num_steps = tf.constant(4, dtype=tf.int32)
    test_groundtruth_text = ['ab', 'abcd']
    decoder_inputs, decoder_inputs_lenghts = label_map_object.text_to_labels(test_groundtruth_text, return_lengths=True)

    output_logits, output_labels, output_lengths = predictor_object.predict(
      test_feature_map,
      decoder_inputs=decoder_inputs,
      decoder_inputs_lengths=decoder_inputs_lenghts,
      num_classes=label_map_object.num_classes,
      go_label=label_map_object.go_label,
      eos_label=label_map_object.eos_label)

    with self.test_session() as sess:
      sess.run([
        tf.global_variables_initializer(),
        tf.tables_initializer()])
      outputs = sess.run({
        'logits': output_logits,
        'labels': output_labels,
        'lengths': output_lengths
      })
      print(outputs)


if __name__ == '__main__':
  tf.test.main()
