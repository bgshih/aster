import functools

import tensorflow as tf

from rare.builders import optimizer_builder
from rare.builders import preprocessor_builder
from rare.core import batcher
from rare.core import preprocessor
from rare.core import standard_fields as fields
from rare.utils import variables_helper
from rare.utils import model_deploy


def _create_input_queue(batch_size_per_clone, create_tensor_dict_fn,
                        batch_queue_capacity, num_batch_queue_threads,
                        prefetch_queue_capacity, data_augmentation_options):
  tensor_dict = create_tensor_dict_fn()
  tensor_dict[fields.InputDataFields.image] = tf.to_float(
    tensor_dict[fields.InputDataFields.image]
  )
  tensor_dict = preprocessor.preprocess(tensor_dict, data_augmentation_options)
  input_queue = batcher.BatchQueue(
    tensor_dict,
    batch_size=batch_size_per_clone,
    batch_queue_capacity=batch_queue_capacity,
    num_batch_queue_threads=num_batch_queue_threads,
    prefetch_queue_capacity=prefetch_queue_capacity
  )
  return input_queue


def _get_inputs(input_queue):
  read_data_list = input_queue.dequeue()
  def _extract_images_and_targets(read_data):
    image = read_data[fields.InputDataFields.image]
    transcript = read_data[fields.InputDataFields.groundtruth_transcript]
    return image, transcript
  return zip(*map(_extract_images_and_targets, read_data_list))


def _create_losses(input_queue, create_model_fn):
  """Creates loss function for a RecognitionModel.
  Args:
    input_queue: BatchQueue object holding enqueued tensor_dicts.
    create_model_fn: A function to create the RecognitionModel.
  """
  model = create_model_fn()

  # get inputs
  images_list, groundtruth_text_list = _get_inputs(input_queue)
  preprocessed_images_list = [model.preprocess(image) for image in images_list]
  images = tf.stack(preprocessed_images_list, axis=0)

  # provide groundtruth
  model.provide_groundtruth(groundtruth_text_list)
  predictions_dict = model.predict(images)

  losses_dict = model.loss(predictions_dict)
  for loss_tensor in losses_dict.values():
    tf.losses.add_loss(loss_tensor)

def train(create_tensor_dict_fn, create_model_fn, train_config, master, task,
          num_clones, worker_replicas, clone_on_cpu, ps_tasks, worker_job_name,
          is_chief, train_dir):
  """Training function for models.
  Args:
    create_tensor_dict_fn: a function to create a tensor input dictionary.
    create_model_fn: a function that creates a DetectionModel and generates
                     losses.
    train_config: a train_pb2.TrainConfig protobuf.
    master: BNS name of the TensorFlow master to use.
    task: The task id of this training instance.
    num_clones: The number of clones to run per machine.
    worker_replicas: The number of work replicas to train with.
    clone_on_cpu: True if clones should be forced to run on CPU.
    ps_tasks: Number of parameter server tasks.
    worker_job_name: Name of the worker job.
    is_chief: Whether this replica is the chief replica.
    train_dir: Directory to write checkpoints and training summaries to.
  """
  model = create_model_fn()
  data_augmentation_options = [
    preprocessor_builder.build(step)
    for step in train_config.data_augmentation_options
  ]

  with tf.Graph().as_default():
    # Build a configuration specifying multi-GPU and multi-replicas.
    deploy_config = model_deploy.DeploymentConfig(
      num_clones=num_clones,
      clone_on_cpu=clone_on_cpu,
      replica_id=task,
      num_replicas=worker_replicas,
      num_ps_tasks=ps_tasks,
      worker_job_name=worker_job_name
    )

    # Place the global step on the device storing the variables.
    with tf.device(deploy_config.variables_device()):
      global_step = tf.train.create_global_step()

    with tf.device(deploy_config.inputs_device()):
      input_queue = _create_input_queue(
        train_config.batch_size // num_clones,
        create_tensor_dict_fn,
        train_config.batch_queue_capacity,
        train_config.num_batch_queue_threads,
        train_config.prefetch_queue_capacity,
        data_augmentation_options
      )

    # Gather initial summaries.
    summaries = set(tf.get_collection(tf.GraphKeys.SUMMARIES))
    global_summaries = set([])

    model_fn = functools.partial(_create_losses, create_model_fn=create_model_fn)
    clones = model_deploy.create_clones(deploy_config, model_fn, [input_queue])
    first_clone_scope = clones[0].scope

    # Gather update_ops from the first clone. These contain, for example,
    # the updates for the batch_norm variables created by model_fn.
    update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS, first_clone_scope)

    with tf.device(deploy_config.optimizer_device()):
      training_optimizer = optimizer_builder.build(
        train_config.optimizer,
        global_summaries
      )

    sync_optimizer = None
    if train_config.sync_replicas:
      training_optimizer = tf.train.SyncReplicasOptimizer(
        training_optimizer,
        replicas_to_aggregate=train_config.replicas_to_aggregate,
        total_num_replicas=train_config.worker_replicas
      )
      sync_optimizer = training_optimizer

    # Create ops required to initialize the model from a given checkpoint.
    init_fn = None
    if train_config.fine_tune_checkpoint:
      var_map = detection_model.restore_map(
        from_detection_checkpoint=train_config.from_detection_checkpoint
      )
      available_var_map = variables_helper.get_variables_available_in_checkpoint(
        var_map,
        train_config.fine_tune_checkpoint
      )
      init_saver = tf.train.Saver(available_var_map)
      def initializer_fn(sess):
        init_saver.restore(sess, train_config.fine_tune_checkpoint)
      init_fn = initializer_fn

    with tf.device(deploy_config.optimizer_device()):
      total_loss, grads_and_vars = model_deploy.optimize_clones(
        clones,
        training_optimizer,
        regularization_losses=None
      )
      total_loss = tf.check_numerics(total_loss, 'LossTensor is inf or nan.')

      # Optionally multiply bias gradients by train_config.bias_grad_multiplier.
      if train_config.bias_grad_multiplier:
        biases_regex_list = ['.*/biases']
        grads_and_vars = variables_helper.multiply_gradients_matching_regex(
          grads_and_vars,
          biases_regex_list,
          multiplier=train_config.bias_grad_multiplier
        )

      # Optionally freeze some layers by setting their gradients to be zero.
      if train_config.freeze_variables:
        grads_and_vars = variables_helper.freeze_gradients_matching_regex(
          grads_and_vars, train_config.freeze_variables
        )

      # Optionally clip gradients
      if train_config.gradient_clipping_by_norm > 0:
        with tf.name_scope('clip_grads'):
          grads_and_vars = tf.contrib.training.clip_gradient_norms(
            grads_and_vars, train_config.gradient_clipping_by_norm
          )

      # Create gradient updates.
      grad_updates = training_optimizer.apply_gradients(
        grads_and_vars,
        global_step=global_step
      )
      update_ops.append(grad_updates)

      update_op = tf.group(*update_ops)
      with tf.control_dependencies([update_op]):
        train_tensor = tf.identity(total_loss, name='train_op')

    # Add summaries.
    for model_var in tf.contrib.framework.get_model_variables():
      global_summaries.add(tf.summary.histogram(model_var.op.name, model_var))
    for loss_tensor in tf.losses.get_losses():
      global_summaries.add(tf.summary.scalar(loss_tensor.op.name, loss_tensor))
    global_summaries.add(
      tf.summary.scalar('TotalLoss', tf.losses.get_total_loss())
    )

    # Add the summaries from the first clone. These contain the summaries
    # created by model_fn and either optimize_clones() or _gather_clone_loss().
    summaries |= set(tf.get_collection(tf.GraphKeys.SUMMARIES,
                                       first_clone_scope))
    summaries |= global_summaries

    # Merge all summaries together.
    summary_op = tf.summary.merge(list(summaries), name='summary_op')

    # Soft placement allows placing on CPU ops without GPU implementation.
    session_config = tf.ConfigProto(allow_soft_placement=True,
                                    log_device_placement=False)

    # Save checkpoints regularly.
    keep_checkpoint_every_n_hours = train_config.keep_checkpoint_every_n_hours
    saver = tf.train.Saver(
      keep_checkpoint_every_n_hours=keep_checkpoint_every_n_hours
    )

    scaffold = tf.train.Scaffold(
      init_fn=init_fn,
      summary_op=summary_op,
      saver=saver
    )
    stop_hook = tf.train.StopAtStepHook(
      num_steps=(train_config.num_steps if train_config.num_steps else None),
    )
    tf.contrib.training.train(
      train_tensor,
      train_dir,
      master=master,
      is_chief=is_chief,
      scaffold=scaffold,
      hooks=[stop_hook],
      chief_only_hooks=None,
      save_checkpoint_secs=train_config.save_checkpoint_secs,
      save_summaries_steps=train_config.save_summaries_steps,
      config=session_config
    )
