import os
import logging

import tensorflow as tf
from tensorflow.python.training import training_util
from tensorflow.python.training import session_run_hook


class ProfileAtStepHook(session_run_hook.SessionRunHook):
  """Hook that requests stop at a specified step."""

  def __init__(self, at_step=None, checkpoint_dir=None, trace_level=tf.RunOptions.FULL_TRACE):
    self._at_step = at_step
    self._do_profile = False
    self._writer = tf.summary.FileWriter(checkpoint_dir)
    self._trace_level = trace_level

  def begin(self):
    self._global_step_tensor = tf.train.get_global_step()
    if self._global_step_tensor is None:
      raise RuntimeError("Global step should be created to use ProfileAtStepHook.")

  def before_run(self, run_context):  # pylint: disable=unused-argument
    if self._do_profile:
      
      options = tf.RunOptions(trace_level=self._trace_level)
    else:
      options = None
    return tf.train.SessionRunArgs(self._global_step_tensor, options=options)

  def after_run(self, run_context, run_values):
    global_step = run_values.results - 1
    if self._do_profile:
      self._do_profile = False
      self._writer.add_run_metadata(run_values.run_metadata,
                                    'trace_{}'.format(global_step), global_step)
      logging.info('Profile trace saved at {}'.format(global_step))
    if global_step == self._at_step:
      self._do_profile = True
