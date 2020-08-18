from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import copy
import os
import time
import collections

import tensorflow.compat.v1 as tf

from object_detection import eval_util
from object_detection import inputs
from object_detection import model_lib
from object_detection.builders import model_builder
from object_detection.builders import optimizer_builder
from object_detection.core import standard_fields as fields
from object_detection.protos import train_pb2
from object_detection.utils import config_util
from object_detection.utils import label_map_util
from object_detection.utils import ops
from object_detection.utils import visualization_utils as vutils
from object_detection import model_lib_v2

import tensorflow as tf
import tensorflow_federated as tff

# pylint: disable=g-import-not-at-top
try:
    from tensorflow.contrib import tpu as contrib_tpu
except ImportError:
    # TF 2.0 doesn't ship with contrib.
    pass
# pylint: enable=g-import-not-at-top

MODEL_BUILD_UTIL_MAP = model_lib.MODEL_BUILD_UTIL_MAP


def train_loop(
        pipeline_config_path,
        model_dir,
        config_override=None,
        train_steps=None,
        use_tpu=False,
        save_final_config=False,
        checkpoint_every_n=5000,
        checkpoint_max_to_keep=7,
        **kwargs):
    """Trains a model using eager + functions.

    This method:
      1. Processes the pipeline configs
      2. (Optionally) saves the as-run config
      3. Builds the model & optimizer
      4. Gets the training input data
      5. Loads a fine-tuning detection or classification checkpoint if requested
      6. Loops over the train data, executing distributed training steps inside
         tf.functions.
      7. Checkpoints the model every `checkpoint_every_n` training steps.
      8. Logs the training metrics as TensorBoard summaries.

    Args:
      pipeline_config_path: A path to a pipeline config file.
      model_dir:
        The directory to save checkpoints and summaries to.
      config_override: A pipeline_pb2.TrainEvalPipelineConfig text proto to
        override the config from `pipeline_config_path`.
      train_steps: Number of training steps. If None, the number of training steps
        is set from the `TrainConfig` proto.
      use_tpu: Boolean, whether training and evaluation should run on TPU.
      save_final_config: Whether to save final config (obtained after applying
        overrides) to `model_dir`.
      checkpoint_every_n:
        Checkpoint every n training steps.
      checkpoint_max_to_keep:
        int, the number of most recent checkpoints to keep in the model directory.
      **kwargs: Additional keyword arguments for configuration override.
    """
    # Parse the configs
    get_configs_from_pipeline_file = MODEL_BUILD_UTIL_MAP[
        'get_configs_from_pipeline_file']
    merge_external_params_with_configs = MODEL_BUILD_UTIL_MAP[
        'merge_external_params_with_configs']
    create_pipeline_proto_from_configs = MODEL_BUILD_UTIL_MAP[
        'create_pipeline_proto_from_configs']

    configs = get_configs_from_pipeline_file(
        pipeline_config_path, config_override=config_override)
    kwargs.update({
        'train_steps': train_steps,
        'use_bfloat16': configs['train_config'].use_bfloat16 and use_tpu
    })
    configs = merge_external_params_with_configs(
        configs, None, kwargs_dict=kwargs)
    model_config = configs['model']
    train_config = configs['train_config']
    train_input_config = configs['train_input_config']

    unpad_groundtruth_tensors = train_config.unpad_groundtruth_tensors
    add_regularization_loss = train_config.add_regularization_loss
    clip_gradients_value = None
    if train_config.gradient_clipping_by_norm > 0:
        clip_gradients_value = train_config.gradient_clipping_by_norm

    # update train_steps from config but only when non-zero value is provided
    if train_steps is None and train_config.num_steps != 0:
        train_steps = train_config.num_steps

    if kwargs['use_bfloat16']:
        tf.compat.v2.keras.mixed_precision.experimental.set_policy(
            'mixed_bfloat16')

    if train_config.load_all_detection_checkpoint_vars:
        raise ValueError('train_pb2.load_all_detection_checkpoint_vars '
                         'unsupported in TF2')

    config_util.update_fine_tune_checkpoint_type(train_config)
    fine_tune_checkpoint_type = train_config.fine_tune_checkpoint_type
    fine_tune_checkpoint_version = train_config.fine_tune_checkpoint_version

    # Write the as-run pipeline config to disk.
    if save_final_config:
        pipeline_config_final = create_pipeline_proto_from_configs(configs)
        config_util.save_pipeline_config(pipeline_config_final, model_dir)

    # Build the model, optimizer, and training input

    ##### Change them for TFF #####

    strategy = tf.compat.v2.distribute.get_strategy()
    with strategy.scope():

        #########
        ##Model##
        #########
        detection_model = model_builder.build(
            model_config=model_config, is_training=True)

        ########
        ##Data##
        ########
        def train_dataset_fn(input_context):
            """Callable to create train input."""
            # Create the inputs.
            train_input = inputs.train_input(
                train_config=train_config,
                train_input_config=train_input_config,
                model_config=model_config,
                model=detection_model,
                input_context=input_context)
            train_input = train_input.repeat()
            return train_input

        train_input = strategy.experimental_distribute_datasets_from_function(
            train_dataset_fn)

        global_step = tf.Variable(
            0, trainable=False, dtype=tf.compat.v2.dtypes.int64, name='global_step',
            aggregation=tf.compat.v2.VariableAggregation.ONLY_FIRST_REPLICA)

        #############
        ##Optimizer##
        #############
        optimizer, (learning_rate,) = optimizer_builder.build(
            train_config.optimizer, global_step=global_step)

        if callable(learning_rate):
            learning_rate_fn = learning_rate
        else:
            def learning_rate_fn(): return learning_rate

    # Train the model
    # Get the appropriate filepath (temporary or not) based on whether the worker
    # is the chief.
    summary_writer_filepath = get_filepath(strategy,
                                           os.path.join(model_dir, 'train'))
    summary_writer = tf.compat.v2.summary.create_file_writer(
        summary_writer_filepath)

    if use_tpu:
        num_steps_per_iteration = 100
    else:
        # TODO(b/135933080) Explore setting to 100 when GPU performance issues
        # are fixed.
        num_steps_per_iteration = 1

    with summary_writer.as_default():
        with strategy.scope():
            with tf.compat.v2.summary.record_if(lambda: global_step % num_steps_per_iteration == 0):
                # Load a fine-tuning checkpoint.
                if train_config.fine_tune_checkpoint:
                    load_fine_tune_checkpoint(detection_model,
                                              train_config.fine_tune_checkpoint,
                                              fine_tune_checkpoint_type,
                                              fine_tune_checkpoint_version,
                                              train_input,
                                              unpad_groundtruth_tensors)

                ckpt = tf.compat.v2.train.Checkpoint(
                    step=global_step, model=detection_model, optimizer=optimizer)

                manager_dir = get_filepath(strategy, model_dir)
                if not strategy.extended.should_checkpoint:
                    checkpoint_max_to_keep = 1
                manager = tf.compat.v2.train.CheckpointManager(
                    ckpt, manager_dir, max_to_keep=checkpoint_max_to_keep)

                # We use the following instead of manager.latest_checkpoint because
                # manager_dir does not point to the model directory when we are running
                # in a worker.
                latest_checkpoint = tf.train.latest_checkpoint(model_dir)
                ckpt.restore(latest_checkpoint)

                def train_step_fn(features, labels):
                    """Single train step."""
                    loss = eager_train_step(
                        detection_model,
                        features,
                        labels,
                        unpad_groundtruth_tensors,
                        optimizer,
                        learning_rate=learning_rate_fn(),
                        add_regularization_loss=add_regularization_loss,
                        clip_gradients_value=clip_gradients_value,
                        global_step=global_step,
                        num_replicas=strategy.num_replicas_in_sync)
                    global_step.assign_add(1)
                    return loss

                def _sample_and_train(strategy, train_step_fn, data_iterator):
                    features, labels = data_iterator.next()
                    logging.debug(features, labels)
                    if hasattr(tf.distribute.Strategy, 'run'):
                        per_replica_losses = strategy.run(
                            train_step_fn, args=(features, labels))
                    else:
                        per_replica_losses = strategy.experimental_run_v2(
                            train_step_fn, args=(features, labels))
                    # TODO(anjalisridhar): explore if it is safe to remove the
                    # num_replicas scaling of the loss and switch this to a ReduceOp.Mean
                    return strategy.reduce(tf.distribute.ReduceOp.SUM,
                                           per_replica_losses, axis=None)

                @tf.function
                def _dist_train_step(data_iterator):
                    """A distributed train step."""

                    if num_steps_per_iteration > 1:
                        for _ in tf.range(num_steps_per_iteration - 1):
                            _sample_and_train(
                                strategy, train_step_fn, data_iterator)

                    return _sample_and_train(strategy, train_step_fn, data_iterator)

                train_input_iter = iter(train_input)

                if int(global_step.value()) == 0:
                    manager.save()

                checkpointed_step = int(global_step.value())
                logged_step = global_step.value()

                last_step_time = time.time()

                for _ in range(global_step.value(), train_steps,
                               num_steps_per_iteration):

                    loss = _dist_train_step(train_input_iter)

                    time_taken = time.time() - last_step_time
                    last_step_time = time.time()

                    tf.compat.v2.summary.scalar(
                        'steps_per_sec', num_steps_per_iteration * 1.0 / time_taken,
                        step=global_step)

                    if global_step.value() - logged_step >= 100:
                        tf.logging.info(
                            'Step {} per-step time {:.3f}s loss={:.3f}'.format(
                                global_step.value(), time_taken / num_steps_per_iteration,
                                loss))
                        logged_step = global_step.value()

                    if ((int(global_step.value()) - checkpointed_step) >=
                            checkpoint_every_n):
                        manager.save()
                        checkpointed_step = int(global_step.value())

    # Remove the checkpoint directories of the non-chief workers that
    # MultiWorkerMirroredStrategy forces us to save during sync distributed
    # training.
    clean_temporary_directories(strategy, manager_dir)
    clean_temporary_directories(strategy, summary_writer_filepath)
