import os
import functools
import logging
import tensorflow as tf

from google.protobuf import text_format
from rare import evaluator
from rare import eval_pb2
from rare import pipeline_pb2
from rare.models import model, model_pb2
from rare.core import input_reader, input_reader_pb2


logging.getLogger('tensorflow').propagate = False # avoid logging duplicates
tf.logging.set_verbosity(tf.logging.INFO)
logging.basicConfig(level=logging.INFO)


flags = tf.app.flags
flags.DEFINE_boolean('repeat', True, 'If true, evaluate repeatedly.')
flags.DEFINE_boolean('eval_training_data', False,
                     'If training data should be evaluated for this job.')
flags.DEFINE_string('checkpoint_dir', '',
                    'Directory containing checkpoints to evaluate, typically '
                    'set to `train_dir` used in the training job.')
flags.DEFINE_string('exp_dir', '',
                    'Directory containing config, training log and evaluations')
flags.DEFINE_string('eval_dir', '',
                    'Directory to write eval summaries to.')
flags.DEFINE_string('pipeline_config_path', '',
                    'Path to a pipeline_pb2.TrainEvalPipelineConfig config '
                    'file. If provided, other configs are ignored')
flags.DEFINE_string('eval_config_path', '',
                    'Path to an eval_pb2.EvalConfig config file.')
flags.DEFINE_string('input_config_path', '',
                    'Path to an input_reader_pb2.InputReader config file.')
flags.DEFINE_string('model_config_path', '',
                    'Path to a model_pb2.DetectionModel config file.')
FLAGS = flags.FLAGS


def get_configs_from_exp_dir():
  pipeline_config_path = os.path.join(FLAGS.exp_dir, 'config/trainval.prototxt')

  pipeline_config = pipeline_pb2.TrainEvalPipelineConfig()
  with tf.gfile.GFile(pipeline_config_path, 'r') as f:
    text_format.Merge(f.read(), pipeline_config)

  model_config = pipeline_config.model
  if FLAGS.eval_training_data:
    eval_config = pipeline_config.train_config
  else:
    eval_config = pipeline_config.eval_config
  input_config = pipeline_config.eval_input_reader

  return model_config, eval_config, input_config


def get_configs_from_pipeline_file():
  """Reads evaluation configuration from a pipeline_pb2.TrainEvalPipelineConfig.

  Reads evaluation config from file specified by pipeline_config_path flag.

  Returns:
    model_config: a model_pb2.DetectionModel
    eval_config: a eval_pb2.EvalConfig
    input_config: a input_reader_pb2.InputReader
  """
  pipeline_config = pipeline_pb2.TrainEvalPipelineConfig()
  with tf.gfile.GFile(FLAGS.pipeline_config_path, 'r') as f:
    text_format.Merge(f.read(), pipeline_config)

  model_config = pipeline_config.model
  if FLAGS.eval_training_data:
    eval_config = pipeline_config.train_config
  else:
    eval_config = pipeline_config.eval_config
  input_config = pipeline_config.eval_input_reader

  return model_config, eval_config, input_config


def get_configs_from_multiple_files():
  """Reads evaluation configuration from multiple config files.

  Reads the evaluation config from the following files:
    model_config: Read from --model_config_path
    eval_config: Read from --eval_config_path
    input_config: Read from --input_config_path

  Returns:
    model_config: a model_pb2.DetectionModel
    eval_config: a eval_pb2.EvalConfig
    input_config: a input_reader_pb2.InputReader
  """
  eval_config = eval_pb2.EvalConfig()
  with tf.gfile.GFile(FLAGS.eval_config_path, 'r') as f:
    text_format.Merge(f.read(), eval_config)

  model_config = model_pb2.DetectionModel()
  with tf.gfile.GFile(FLAGS.model_config_path, 'r') as f:
    text_format.Merge(f.read(), model_config)

  input_config = input_reader_pb2.InputReader()
  with tf.gfile.GFile(FLAGS.input_config_path, 'r') as f:
    text_format.Merge(f.read(), input_config)

  return model_config, eval_config, input_config


def main(unused_argv):
  if FLAGS.exp_dir:
    checkpoint_dir = os.path.join(FLAGS.exp_dir, 'log')
    eval_dir = os.path.join(FLAGS.exp_dir, 'log/eval')
    model_config, eval_config, input_config = get_configs_from_exp_dir()
  else:
    assert FLAGS.checkpoint_dir, '`checkpoint_dir` is missing.'
    assert FLAGS.eval_dir, '`eval_dir` is missing.'
    if FLAGS.pipeline_config_path:
      model_config, eval_config, input_config = get_configs_from_pipeline_file()
    else:
      model_config, eval_config, input_config = get_configs_from_multiple_files()
    checkpoint_dir = FLAGS.checkpoint_dir
    eval_dir = FLAGS.eval_dir

  model_fn = functools.partial(
      model.build,
      model_config=model_config,
      is_training=False)

  create_input_dict_fn = functools.partial(
      input_reader.build,
      input_config)

  evaluator.evaluate(create_input_dict_fn, model_fn, eval_config,
                     checkpoint_dir, eval_dir,
                     repeat_evaluation=FLAGS.repeat)

if __name__ == '__main__':
  tf.app.run()
