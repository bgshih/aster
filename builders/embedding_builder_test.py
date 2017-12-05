import tensorflow as tf

from google.protobuf import text_format
from rare.builders import embedding_builder
from rare.protos import embedding_pb2


class EmbeddingBuilderTest(tf.test.TestCase):

  def test_build_embedding(self):
    embedding_text_proto = """
    one_hot_embedding {
    }
    """
    embedding_proto = embedding_pb2.Embedding()
    text_format.Merge(embedding_text_proto, embedding_proto)
    embedding_object = embedding_builder.build(embedding_proto)

if __name__ == '__main__':
  tf.test.main()
