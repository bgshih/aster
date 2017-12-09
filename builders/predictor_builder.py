import tensorflow as tf

from rare.predictors import attention_predictor
from rare.protos import predictor_pb2
from rare.builders import rnn_cell_builder
from rare.builders import embedding_builder

def build(config,
          label_map=None,
          is_training=None):
  if not isinstance(config, predictor_pb2.Predictor):
    raise ValueError('config not of type '
                     'predictor_pb2.predictor')
  predictor_oneof = config.WhichOneof('predictor_oneof')

  if predictor_oneof == 'attention_predictor':
    attention_predictor_config = config.attention_predictor

    rnn_cell_object = rnn_cell_builder.build(attention_predictor_config.rnn_cell)
    embedding_object = embedding_builder.build(attention_predictor_config.output_embedding)
    attention_predictor_object = attention_predictor.AttentionPredictor(
      rnn_cell=rnn_cell_object,
      num_attention_units=attention_predictor_config.num_attention_units,
      attention_conv_kernel_size=attention_predictor_config.attention_conv_kernel_size,
      output_embedding=
      is_training=is_training
    )
    return attention_predictor_object

  else:
    raise ValueError('Unknown predictor_oneof: {}'.format(predictor_oneof))
