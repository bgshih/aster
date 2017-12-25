from rare.builders import hyperparams_builder
from rare.protos import convnet_pb2
from rare.convnets import crnn_net


def build(config, is_training):
  if not isinstance(config, convnet_pb2.Convnet):
    raise ValueError('config not of type convnet_pb2.Convnet')
  convnet_oneof = config.WhichOneof('convnet_oneof')
  if convnet_oneof == 'crnn_net':
    return _build_crnn_net(config.crnn_net, is_training)
  elif convnet_oneof == 'res_net':
    return _build_res_net(config.resnet, is_training)
  else:
    raise ValueError('Unknown convnet_oneof: {}'.format(convnet_oneof))

def _build_crnn_net(config, is_training):
  if not isinstance(config, convnet_pb2.CrnnNet):
    raise ValueError('config is not of type convnet_pb2.CrnnNet')

  if config.net_type == convnet_pb2.CrnnNet.SINGLE_BRANCH:
    crnn_net_class = crnn_net.CrnnNet
  elif config.net_type == convnet_pb2.CrnnNet.TWO_BRANCHES:
    crnn_net_class = crnn_net.CrnnNetTwoBranches
  elif config.net_type == convnet_pb2.CrnnNet.THREE_BRANCHES:
    crnn_net_class = crnn_net.CrnnNetThreeBranches
  else:
    raise ValueError('Unknown net_type: {}'.format(config.net_type))

  hyperparams_object = hyperparams_builder.build(config.conv_hyperparams, is_training)

  return crnn_net_class(
    conv_hyperparams=hyperparams_object,
    summarize_activations=config.summarize_activations,
    is_training=is_training)

def _build_res_net(config, is_training):
  raise NotImplementedError
