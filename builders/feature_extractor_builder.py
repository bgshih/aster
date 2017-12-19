import tensorflow as tf

from rare.protos import feature_extractor_pb2
from rare.builders import hyperparams_builder
from rare.feature_extractors import baseline_feature_extractor
from rare.feature_extractors import resnet_feature_extractor


def build(config, is_training):
  if not isinstance(config, feature_extractor_pb2.FeatureExtractor):
    raise ValueError('config not of type '
                     'feature_extractor_pb2.FeatureExtractor')

  feature_extractor_oneof = config.WhichOneof('feature_extractor_oneof')
  if feature_extractor_oneof == 'baseline_feature_extractor':
    baseline_feature_extractor_config = config.baseline_feature_extractor
    conv_hyperparams = hyperparams_builder.build(
      baseline_feature_extractor_config.conv_hyperparams,
      is_training)
    return baseline_feature_extractor.BaselineFeatureExtractor(
      conv_hyperparams=conv_hyperparams,
      summarize_inputs=baseline_feature_extractor_config.summarize_inputs)

  elif feature_extractor_oneof == 'resnet_feature_extractor':
    resnet_config = config.resnet_feature_extractor

    resnet_type = resnet_config.resnet_type
    if resnet_type == feature_extractor_pb2.ResnetFeatureExtractor.RESNET_50:
      resnet_class = resnet_feature_extractor.Resnet52LayerFeatureExtractor
    else:
      raise ValueError('Unknown resnet type: {}'.format(resnet_type))

    conv_hyperparams = hyperparams_builder.build(
      resnet_config.conv_hyperparams,
      is_training)
    return resnet_class(
      conv_hyperparams=conv_hyperparams,
      summarize_inputs=resnet_config.summarize_inputs,
      is_training=is_training)

  else:
    raise ValueError('Unknown feature_extractor_oneof: {}'.format(feature_extractor_oneof))
  