import tensorflow as tf
from seq2seq.attention_wrapper import *


class AttentionWrapperSyncAttention(AttentionWrapper):

  def call(self, inputs, state):
    if not isinstance(state, AttentionWrapperState):
      raise TypeError("Expected state to be instance of AttentionWrapperState. "
                      "Received type %s instead."  % type(state))

    