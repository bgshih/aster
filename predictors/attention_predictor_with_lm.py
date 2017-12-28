import logging
import functools

import tensorflow as tf
from tensorflow.contrib import seq2seq
from tensorflow.contrib import rnn
from tensorflow.python.ops import variable_scope as vs
from tensorflow.python.util import nest

from rare.predictors import attention_predictor


class AttentionPredictorWithLanguageModel(attention_predictor.AttentionPredictor):
  """Attention decoder coupled with a language model RNN."""

  def __init__(self,
               rnn_cell=None,
               rnn_regularizer=None,
               lm_rnn_cell=None,
               num_attention_units=None,
               max_num_steps=None,
               multi_attention=False,
               beam_width=None,
               reverse=False,
               label_map=None,
               loss=None,
               is_training=True):
    super(AttentionPredictorWithLanguageModel, self).__init__(
      rnn_cell=rnn_cell,
      rnn_regularizer=rnn_regularizer,
      num_attention_units=num_attention_units,
      max_num_steps=max_num_steps,
      multi_attention=multi_attention,
      beam_width=beam_width,
      reverse=reverse,
      label_map=label_map,
      loss=loss,
      is_training=is_training
    )
    self._lm_rnn_cell = lm_rnn_cell

  def _build_decoder_cell(self, feature_maps):
    attention_mechanism = self._build_attention_mechanism(feature_maps)
    attention_decoder_cell = seq2seq.AttentionWrapper(
      self._rnn_cell,
      attention_mechanism,
      output_attention=False)
    decoder_cell = ConcatOutputMultiRNNCell([attention_decoder_cell, self._lm_rnn_cell])
    return decoder_cell


class ConcatOutputMultiRNNCell(rnn.MultiRNNCell):
  """RNN cell composed of multiple RNN cells whose outputs are concatenated along depth."""

  @property
  def output_size(self):
    return sum([cell.output_size for cell in self._cells])

  def call(self, inputs, state):
    cur_state_pos = 0
    cur_inp = inputs
    outputs = []
    new_states = []
    for i, cell in enumerate(self._cells):
      with vs.variable_scope("cell_%d" % i):
        if self._state_is_tuple:
          if not nest.is_sequence(state):
            raise ValueError(
                "Expected state to be a tuple of length %d, but received: %s" %
                (len(self.state_size), state))
          cur_state = state[i]
        else:
          cur_state = array_ops.slice(state, [0, cur_state_pos],
                                      [-1, cell.state_size])
          cur_state_pos += cell.state_size
        cur_output, new_state = cell(cur_inp, cur_state)
        new_states.append(new_state)
        outputs.append(cur_output)

    new_states = (tuple(new_states) if self._state_is_tuple else
                  array_ops.concat(new_states, 1))
    output = tf.concat(outputs, -1)

    return output, new_states
