import tensorflow as tf

from rare.protos import feature_extractor_pb2
from rare.feature_extractors import baseline_feature_extractor


def build(config, is_training):
  if not isinstance(config, feature_extractor_pb2.FeatureExtractor):
    raise ValueError('config not of type '
                     'feature_extractor_pb2.FeatureExtractor')

  feature_extractor_oneof = config.WhichOneof('feature_extractor_oneof')
  if feature_extractor_oneof == 'baseline_feature_extractor':
    baseline_feature_extractor_config = config.baseline_feature_extractor
    return baseline_feature_extractor.BaselineFeatureExtractor()
  else:
    raise ValueError('Unknown feature_extractor_oneof: {}'.format(feature_extractor_oneof))
  