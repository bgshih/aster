import tensorflow as tf

from rare.protos import feature_extractor_pb2
from rare.builders import hyperparams_builder
from rare.feature_extractors import baseline_feature_extractor


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
  else:
    raise ValueError('Unknown feature_extractor_oneof: {}'.format(feature_extractor_oneof))
  