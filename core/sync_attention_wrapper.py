from tensorflow.python.ops import array_ops
from tensorflow.contrib import seq2seq
from tensorflow.contrib.seq2seq.python.ops.attention_wrapper import _compute_attention


class SyncAttentionWrapper(seq2seq.AttentionWrapper):
  
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
      attention, alignments = _compute_attention(
          attention_mechanism, state.cell_state, previous_alignments[i],
          self._attention_layers[i] if self._attention_layers else None)
      alignment_history = previous_alignment_history[i].write(
          state.time, alignments) if self._alignment_history else ()

      all_alignments.append(alignments)
      all_histories.append(alignment_history)
      all_attentions.append(attention)

    attention = array_ops.concat(all_attentions, 1)

    cell_inputs = self._cell_input_fn(inputs, attention)
    cell_output, next_cell_state = self._cell(cell_inputs, state.cell_state)

    next_state = AttentionWrapperState(
        time=state.time + 1,
        cell_state=next_cell_state,
        attention=attention,
        alignments=self._item_or_tuple(all_alignments),
        alignment_history=self._item_or_tuple(all_histories))
    
    if self._output_attention:
      return attention, next_state
    else:
      return cell_output, next_state
