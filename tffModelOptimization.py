import functools

import numpy as np
import tensorflow as tf
import tensorflow_federated as tff
import os
import shutil
from tensorflow_model_optimization.python.core.internal import tensor_encoding as te

@tff.federated_computation
def hello_world():
  return 'Hello, World!'

print(hello_world())


# This value only applies to EMNIST dataset, consider choosing appropriate
# values if switching to other datasets.
MAX_CLIENT_DATASET_SIZE = 418

CLIENT_EPOCHS_PER_ROUND = 1
CLIENT_BATCH_SIZE = 20
TEST_BATCH_SIZE = 500

emnist_train, emnist_test = tff.simulation.datasets.emnist.load_data(
    only_digits=True)

def reshape_emnist_element(element):
  return (tf.expand_dims(element['pixels'], axis=-1), element['label'])

def preprocess_train_dataset(dataset):
  """Preprocessing function for the EMNIST training dataset."""
  return (dataset
          # Shuffle according to the largest client dataset
          .shuffle(buffer_size=MAX_CLIENT_DATASET_SIZE)
          # Repeat to do multiple local epochs
          .repeat(CLIENT_EPOCHS_PER_ROUND)
          # Batch to a fixed client batch size
          .batch(CLIENT_BATCH_SIZE, drop_remainder=False)
          # Preprocessing step
          .map(reshape_emnist_element))

emnist_train = emnist_train.preprocess(preprocess_train_dataset)


def create_original_fedavg_cnn_model(only_digits=True):
  """The CNN model used in https://arxiv.org/abs/1602.05629."""
  data_format = 'channels_last'

  max_pool = functools.partial(
      tf.keras.layers.MaxPooling2D,
      pool_size=(2, 2),
      padding='same',
      data_format=data_format)
  conv2d = functools.partial(
      tf.keras.layers.Conv2D,
      kernel_size=5,
      padding='same',
      data_format=data_format,
      activation=tf.nn.relu)

  model = tf.keras.models.Sequential([
      tf.keras.layers.InputLayer(input_shape=(28, 28, 1)),
      conv2d(filters=32),
      max_pool(),
      conv2d(filters=64),
      max_pool(),
      tf.keras.layers.Flatten(),
      tf.keras.layers.Dense(512, activation=tf.nn.relu),
      tf.keras.layers.Dense(10 if only_digits else 62),
      tf.keras.layers.Softmax(),
  ])

  return model

# Gets the type information of the input data. TFF is a strongly typed
# functional programming framework, and needs type information about inputs to 
# the model.
input_spec = emnist_train.create_tf_dataset_for_client(
    emnist_train.client_ids[0]).element_spec

def tff_model_fn():
  keras_model = create_original_fedavg_cnn_model()
  return tff.learning.from_keras_model(
      keras_model=keras_model,
      input_spec=input_spec,
      loss=tf.keras.losses.SparseCategoricalCrossentropy(),
      metrics=[tf.keras.metrics.SparseCategoricalAccuracy()])

federated_averaging = tff.learning.build_federated_averaging_process(
    model_fn=tff_model_fn,
    client_optimizer_fn=lambda: tf.keras.optimizers.SGD(learning_rate=0.02),
    server_optimizer_fn=lambda: tf.keras.optimizers.SGD(learning_rate=1.0))

def format_size(size):
  """A helper function for creating a human-readable size."""
  size = float(size)
  for unit in ['B','KiB','MiB','GiB']:
    if size < 1024.0:
      return "{size:3.2f}{unit}".format(size=size, unit=unit)
    size /= 1024.0
  return "{size:.2f}{unit}".format(size=size, unit='TiB')

def set_sizing_environment():
  """Creates an environment that contains sizing information."""
  # Creates a sizing executor factory to output communication cost
  # after the training finishes. Note that sizing executor only provides an
  # estimate (not exact) of communication cost, and doesn't capture cases like
  # compression of over-the-wire representations. However, it's perfect for
  # demonstrating the effect of compression in this tutorial.
  sizing_factory = tff.framework.sizing_executor_factory()

  # TFF has a modular runtime you can configure yourself for various
  # environments and purposes, and this example just shows how to configure one
  # part of it to report the size of things.
  context = tff.framework.ExecutionContext(executor_fn=sizing_factory)
  tff.framework.set_default_context(context)

  return sizing_factory

def train(federated_averaging_process, num_rounds, num_clients_per_round, summary_writer):
  """Trains the federated averaging process and output metrics."""
  # Create a environment to get communication cost.
  environment = set_sizing_environment()

  # Initialize the Federated Averaging algorithm to get the initial server state.
  state = federated_averaging_process.initialize()

  with summary_writer.as_default():
    for round_num in range(num_rounds):
      # Sample the clients parcitipated in this round.
      sampled_clients = np.random.choice(
          emnist_train.client_ids,
          size=num_clients_per_round,
          replace=False)
      # Create a list of `tf.Dataset` instances from the data of sampled clients.
      sampled_train_data = [
          emnist_train.create_tf_dataset_for_client(client)
          for client in sampled_clients
      ]
      # Round one round of the algorithm based on the server state and client data
      # and output the new state and metrics.
      state, metrics = federated_averaging_process.next(state, sampled_train_data)

      # For more about size_info, please see https://www.tensorflow.org/federated/api_docs/python/tff/framework/SizeInfo
      size_info = environment.get_size_info()
      broadcasted_bits = size_info.broadcast_bits[-1]
      aggregated_bits = size_info.aggregate_bits[-1]

      print('round {:2d}, metrics={}, broadcasted_bits={}, aggregated_bits={}'.format(round_num, metrics, format_size(broadcasted_bits), format_size(aggregated_bits)))

      # Add metrics to Tensorboard.
      for name, value in metrics['train'].items():
          tf.summary.scalar(name, value, step=round_num)

      # Add broadcasted and aggregated data size to Tensorboard.
      tf.summary.scalar('cumulative_broadcasted_bits', broadcasted_bits, step=round_num)
      tf.summary.scalar('cumulative_aggregated_bits', aggregated_bits, step=round_num)
      summary_writer.flush()

logdir = os.path.join("logs")
if os.path.exists(logdir):
    shutil.rmtree(logdir)
os.mkdir(logdir)
summary_writer = tf.summary.create_file_writer(logdir)


import grpc

ip_address = '10.0.0.110'  
port = 802 

client_ex = []
for i in range(10):
  channel = grpc.insecure_channel('{}:{}'.format(ip_address, port))
  client_ex.append(tff.framework.RemoteExecutor(channel, rpc_mode='STREAMING'))

factory = tff.framework.worker_pool_executor_factory(client_ex)
context = tff.framework.ExecutionContext(factory)
tff.framework.set_default_context(context)

train(federated_averaging_process=federated_averaging, num_rounds=10,
      num_clients_per_round=10, summary_writer=summary_writer)




def broadcast_encoder_fn(value):
  """Function for building encoded broadcast."""
  spec = tf.TensorSpec(value.shape, value.dtype)
  if value.shape.num_elements() > 10000:
    return te.encoders.as_simple_encoder(
        te.encoders.uniform_quantization(bits=8), spec)
  else:
    return te.encoders.as_simple_encoder(te.encoders.identity(), spec)


def mean_encoder_fn(value):
  """Function for building encoded mean."""
  spec = tf.TensorSpec(value.shape, value.dtype)
  if value.shape.num_elements() > 10000:
    return te.encoders.as_gather_encoder(
        te.encoders.uniform_quantization(bits=8), spec)
  else:
    return te.encoders.as_gather_encoder(te.encoders.identity(), spec)
"""
encoded_broadcast_process = (
    tff.learning.framework.build_encoded_broadcast_process_from_model(
        tff_model_fn, broadcast_encoder_fn))
encoded_mean_process = (
    tff.learning.framework.build_encoded_mean_process_from_model(
    tff_model_fn, mean_encoder_fn))

federated_averaging_with_compression = tff.learning.build_federated_averaging_process(
    tff_model_fn,
    client_optimizer_fn=lambda: tf.keras.optimizers.SGD(learning_rate=0.02),
    server_optimizer_fn=lambda: tf.keras.optimizers.SGD(learning_rate=1.0),
    broadcast_process=encoded_broadcast_process,
    aggregation_process=encoded_mean_process)





logdir = os.path.join("complogs")
if os.path.exists(logdir):
    shutil.rmtree(logdir)
os.mkdir(logdir)
summary_writer_for_compression = tf.summary.create_file_writer(logdir)
train(federated_averaging_process=federated_averaging_with_compression, 
      num_rounds=10,
      num_clients_per_round=10,
      summary_writer=summary_writer_for_compression)

"""