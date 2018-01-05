import os

import tensorflow as tf
from tensorflow.python.training import training_util
from tensorflow.python.training import basic_session_run_hook
from tensorflow.python.training import session_run_hook


class ProfileAtStepHook(basic_session_run_hook.StopAtStepHook):

  def __init__(self, at_step=None, output_path=None):
    super(ProfileAtStepHook)
    self._at_step = at_step
    self._output_path = output_path

    if not os.path.exists(output_path):
      raise ValueError('Output path not exist: {}'.format(output_path))
  
  def before_run(self, run_context):
    if 
    session_run_args = session_run_hook.SessionRunArgs(
      [],
      options=tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
    )

  def after_run(self, run_context, run_values):
    fetched_timeline = run_values.run_metadata.step_stats
    chrome_trace = fetched_timeline.generate_chrome_trace_format()
    with 


class ProfileAtStepHook(session_run_hook.SessionRunHook):
  """Hook that requests stop at a specified step."""

  def __init__(self, at_step=None):
    self._at_step = at_step

  def begin(self):
    self._global_step_tensor = training_util.get_global_step()
    if self._global_step_tensor is None:
      raise RuntimeError("Global step should be created to use StopAtStepHook.")

  def after_create_session(self, session, coord):
    global_step = session.run(self._global_step_tensor)

  def before_run(self, run_context):  # pylint: disable=unused-argument
    return SessionRunArgs(self._global_step_tensor)

  def after_run(self, run_context, run_values):
    global_step = run_values.results
    if global_step == self._at_step:
      run_context.request_stop()
