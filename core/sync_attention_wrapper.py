from tensorflow.python.ops import array_ops
from tensorflow.contrib import rnn
from tensorflow.contrib import seq2seq
from tensorflow.contrib.seq2seq.python.ops.attention_wrapper import _compute_attention


class SyncAttentionWrapper(seq2seq.AttentionWrapper):

  def __init__(self,
               cell,
               attention_mechanism,
               attention_layer_size=None,
               alignment_history=False,
               cell_input_fn=None,
               output_attention=True,
               initial_cell_state=None,
               name=None):
    if not isinstance(cell, (rnn.LSTMCell, rnn.GRUCell)):
      raise ValueError('SyncAttentionWrapper only supports LSTMCell and GRUCell, '
                       'Got: {}'.format(cell))
    super(SyncAttentionWrapper, self).__init__(
      cell,
      attention_mechanism,
      attention_layer_size=attention_layer_size,
      alignment_history=alignment_history,
      cell_input_fn=cell_input_fn,
      output_attention=output_attention,
      initial_cell_state=initial_cell_state,
      name=name
    )
  
  def call(self, inputs, state):
    if not isinstance(state, seq2seq.AttentionWrapperState):
      raise TypeError("Expected state to be instance of AttentionWrapperState. "
                      "Received type %s instead."  % type(state))

    if self._is_multi:
      previous_alignments = state.alignments
      previous_alignment_history = state.alignment_history
    else:
      previous_alignments = [state.alignments]
      previous_alignment_history = [state.alignment_history]

    all_alignments = []
    all_attentions = []
    all_histories = []
    for i, attention_mechanism in enumerate(self._attention_mechanisms):
      if isinstance(self._cell, rnn.LSTMCell):
        rnn_cell_state = state.cell_state.h
      else:
        rnn_cell_state = state.cell_state
      attention, alignments = _compute_attention(
          attention_mechanism, rnn_cell_state, previous_alignments[i],
          self._attention_layers[i] if self._attention_layers else None)
      alignment_history = previous_alignment_history[i].write(
          state.time, alignments) if self._alignment_history else ()

      all_alignments.append(alignments)
      all_histories.append(alignment_history)
      all_attentions.append(attention)

    attention = array_ops.concat(all_attentions, 1)

    cell_inputs = self._cell_input_fn(inputs, attention)
    cell_output, next_cell_state = self._cell(cell_inputs, state.cell_state)

    next_state = seq2seq.AttentionWrapperState(
        time=state.time + 1,
        cell_state=next_cell_state,
        attention=attention,
        alignments=self._item_or_tuple(all_alignments),
        alignment_history=self._item_or_tuple(all_histories))
    
    if self._output_attention:
      return attention, next_state
    else:
      return cell_output, next_state
