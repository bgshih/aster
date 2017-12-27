import logging
import tensorflow as tf

from rare.core import preprocessor
from rare.core import prefetcher
from rare.core import standard_fields as fields
from rare.builders import preprocessor_builder
from rare import eval_util


EVAL_METRICS_FN_DICT = {
  'recognition_metrics': eval_util.evaluate_recognition_results,
}


def _extract_prediction_tensors(model,
                                create_input_dict_fn,
                                data_preprocessing_steps,
                                ignore_groundtruth=False):
  # input queue
  input_dict = create_input_dict_fn()
  prefetch_queue = prefetcher.prefetch(input_dict, capacity=500)
  input_dict = prefetch_queue.dequeue()
  original_image = input_dict[fields.InputDataFields.image]
  original_image_shape = tf.shape(original_image)

  # data preprocessing
  preprocessed_input_dict = preprocessor.preprocess(input_dict, data_preprocessing_steps)

  # model inference
  preprocessed_image = preprocessed_input_dict[fields.InputDataFields.image]
  preprocessed_image_shape = tf.shape(preprocessed_image)
  predictions_dict = model.predict(
      model.preprocess(
          tf.to_float(
              tf.expand_dims(preprocessed_image, 0))))
  recognitions = model.postprocess(predictions_dict)

  tensor_dict = {
    'original_image': original_image,
    'original_image_shape': original_image_shape,
    'preprocessed_image_shape': preprocessed_image_shape,
    'filename': preprocessed_input_dict[fields.InputDataFields.filename],
    'groundtruth_text': input_dict[fields.InputDataFields.groundtruth_text],
    'recognition_text': recognitions['text'][0]
  }
  return tensor_dict


def evaluate(create_input_dict_fn, create_model_fn, eval_config,
             checkpoint_dir, eval_dir,
             repeat_evaluation=True):
  model = create_model_fn()
  data_preprocessing_steps = [
      preprocessor_builder.build(step)
      for step in eval_config.data_preprocessing_steps]

  tensor_dict = _extract_prediction_tensors(
      model=model,
      create_input_dict_fn=create_input_dict_fn,
      data_preprocessing_steps=data_preprocessing_steps,
      ignore_groundtruth=eval_config.ignore_groundtruth)

  def _process_batch(tensor_dict, sess, batch_index, counters, update_op):
    if batch_index >= eval_config.num_visualizations:
      if 'original_image' in tensor_dict:
        tensor_dict = {k: v for (k, v) in tensor_dict.items()
                       if k != 'original_image'}
    try:
      (result_dict, _) = sess.run([tensor_dict, update_op])
      counters['success'] += 1
    except tf.errors.InvalidArgumentError:
      logging.info('Skipping image')
      counters['skipped'] += 1
      return {}
    global_step = tf.train.global_step(sess, tf.train.get_global_step())
    if batch_index < eval_config.num_visualizations:
      raise NotImplementedError

      eval_util.print_recognition_results(
          result_dict,
          tag,
          global_step,
          summary_dir=eval_dir,
          export_dir=eval_config.visualization_export_dir,
          show_groundtruth=False,
          show_segments_and_links=False)

    return result_dict

  def _process_aggregated_results(result_lists):
    eval_metric_fn_key = eval_config.metrics_set
    if eval_metric_fn_key not in EVAL_METRICS_FN_DICT:
      raise ValueError('Metric not found: {}'.format(eval_metric_fn_key))
    return EVAL_METRICS_FN_DICT[eval_metric_fn_key](result_lists)

  variables_to_restore = tf.global_variables()
  global_step = tf.train.get_or_create_global_step()
  variables_to_restore.append(global_step)
  if eval_config.use_moving_averages:
    variable_averages = tf.train.ExponentialMovingAverage(0.0)
    variables_to_restore = variable_averages.variables_to_restore()
  saver = tf.train.Saver(variables_to_restore)
  def _restore_latest_checkpoint(sess):
    latest_checkpoint = tf.train.latest_checkpoint(checkpoint_dir)
    saver.restore(sess, latest_checkpoint)

  eval_util.repeated_checkpoint_run(
      tensor_dict=tensor_dict,
      update_op=tf.no_op(),
      summary_dir=eval_dir,
      aggregated_result_processor=_process_aggregated_results,
      batch_processor=_process_batch,
      checkpoint_dirs=[checkpoint_dir],
      variables_to_restore=None,
      restore_fn=_restore_latest_checkpoint,
      num_batches=eval_config.num_examples,
      eval_interval_secs=eval_config.eval_interval_secs,
      max_number_of_evaluations=(
          1 if eval_config.ignore_groundtruth else
          eval_config.max_evals if eval_config.max_evals else
          None if repeat_evaluation else 1),
      master=eval_config.eval_master,
      save_graph=eval_config.save_graph,
      save_graph_dir=(eval_dir if eval_config.save_graph else ''))
