import tensorflow as tf

from rare.core import embedding
from rare.protos import embedding_pb2

def build(embedding_config):
  if not isinstance(embedding_config, embedding_pb2.Embedding):
    raise ValueError('embedding_config not of type '
                     'embedding_pb2.Embedding')

  embedding_oneof = embedding_config.WhichOneof('embedding_oneof')
  if embedding_oneof == 'one_hot_embedding':
    one_hot_embedding_config = embedding_config.one_hot_embedding
    return embedding.OneHotEmbedding()
  else:
    raise ValueError('Unknown embedding_oneof: {}'.format(embedding_oneof))