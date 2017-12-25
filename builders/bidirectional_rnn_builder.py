from rare.core import bidirectional_rnn
from rare.protos import hyperparams_pb2
from rare.protos import bidirectional_rnn_pb2
from rare.builders import hyperparams_builder
from rare.builders import rnn_cell_builder


def build(config, is_training):
  if not isinstance(config, bidirectional_rnn_pb2.BidirectionalRnn):
    raise ValueError('config not of type bidirectional_rnn_pb2.BidirectionalRnn')

  if config.static:
    brnn_class = bidirectional_rnn.StaticBidirectionalRnn
  else:
    brnn_class = bidirectional_rnn.DynamicBidirectionalRnn

  fw_cell_object = rnn_cell_builder.build(config.fw_bw_rnn_cell)
  bw_cell_object = rnn_cell_builder.build(config.fw_bw_rnn_cell)
  rnn_regularizer_object = hyperparams_builder._build_regularizer(config.rnn_regularizer)
  fc_hyperparams_object = None
  if config.num_output_units:
    if config.fc_hyperparams.op != hyperparams_pb2.Hyperparams.FC:
      raise ValueError('op type must be FC')
    fc_hyperparams_object = hyperparams_builder.build(config.fc_hyperparams, is_training)

  return brnn_class(
    fw_cell_object, bw_cell_object,
    rnn_regularizer=rnn_regularizer_object,
    num_output_units=config.num_output_units,
    fc_hyperparams=fc_hyperparams_object,
    summarize_activations=config.summarize_activations)
