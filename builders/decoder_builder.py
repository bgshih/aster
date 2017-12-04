from rare.decoders import AttentionDecoder
from rare.protos import decoder_pb2
from rare.builders import rnn_cell_builder

def build(decoder_config):
  if not isinstance(decoder_config, decoder_pb2.Decoder):
    raise ValueError('decoder_config not of type '
                     'decoder_pb2.Decoder')
  decoder_oneof = decoder_config.WhichOneof('decoder_oneof')

  if decoder_oneof == 'attention_decoder':
    attention_decoder_config = decoder_config.attention_decoder

    rnn_cell = rnn_cell_builder.build(attention_decoder_config.rnn_cell)
    num_attention_units = attention_decoder_config.num_attention_units
    attention_conv_kernel_size = attention_decoder_config.attention_conv_kernel_size

    attention_decoder_object = AttentionDecoder(
      rnn_cell,
      num_attention_units,
      attention_conv_kernel_size
    )
    return attention_decoder_object

  else:
    raise ValueError('Unknown decoder_oneof: {}'.format(decoder_oneof))
