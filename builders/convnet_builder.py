from rare.builders import hyperparams_builder
from rare.protos import convnet_pb2
from rare.convnets import crnn_net
from rare.convnets import resnet
from rare.convnets import stn_convnet


def build(config, is_training):
  if not isinstance(config, convnet_pb2.Convnet):
    raise ValueError('config not of type convnet_pb2.Convnet')
  convnet_oneof = config.WhichOneof('convnet_oneof')
  if convnet_oneof == 'crnn_net':
    return _build_crnn_net(config.crnn_net, is_training)
  elif convnet_oneof == 'resnet':
    return _build_resnet(config.resnet, is_training)
  elif convnet_oneof == 'stn_resnet':
    return _build_stn_resnet(config.stn_resnet, is_training)
  elif convnet_oneof == 'stn_convnet':
    return _build_stn_convnet(config.stn_convnet, is_training)
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

def _build_resnet(config, is_training):
  if not isinstance(config, convnet_pb2.ResNet):
    raise ValueError('config is not of type convnet_pb2.ResNet')

  if config.net_type != convnet_pb2.ResNet.SINGLE_BRANCH:
    raise ValueError('Only SINGLE_BRANCH is supported for ResNet')

  resnet_depth = config.net_depth
  if resnet_depth == convnet_pb2.ResNet.RESNET_50:
    resnet_class = resnet.Resnet50Layer
  else:
    raise ValueError('Unknown resnet depth: {}'.format(resnet_depth))

  conv_hyperparams = hyperparams_builder.build(config.conv_hyperparams, is_training)
  return resnet_class(
    conv_hyperparams=conv_hyperparams,
    summarize_activations=config.summarize_activations,
    is_training=is_training,
  )

def _build_stn_resnet(config, is_training):
  if not isinstance(config, convnet_pb2.StnResnet):
    raise ValueError('config is not of type convnet_pb2.StnResnet')
  return resnet.ResnetForSTN(
    conv_hyperparams=hyperparams_builder.build(config.conv_hyperparams, is_training),
    summarize_activations=config.summarize_activations,
    is_training=is_training
  )

def _build_stn_convnet(config, is_training):
  if not isinstance(config, convnet_pb2.StnConvnet):
    raise ValueError('config is not of type convnet_pb2.StnConvnet')
  return stn_convnet.StnConvnet(
    conv_hyperparams=hyperparams_builder.build(config.conv_hyperparams, is_training),
    summarize_activations=config.summarize_activations,
    is_training=is_training
  )
