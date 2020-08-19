import collections
import time

import tensorflow as tf
import tensorflow_federated as tff

source, _ = tff.simulation.datasets.emnist.load_data()


def map_fn(example):
  return collections.OrderedDict(
      x=tf.reshape(example['pixels'], [-1, 784]), y=example['label'])


def client_data(n):
  ds = source.create_tf_dataset_for_client(source.client_ids[n])
  return ds.repeat(10).batch(20).map(map_fn)


train_data = [client_data(n) for n in range(40)]
input_spec = train_data[0].element_spec


def model_fn():
  model = tf.keras.models.Sequential([
      tf.keras.layers.Input(shape=(784,)),
      tf.keras.layers.Dense(units=10, kernel_initializer='zeros'),
      tf.keras.layers.Softmax(),
  ])
  return tff.learning.from_keras_model(
      model,
      input_spec=input_spec,
      loss=tf.keras.losses.SparseCategoricalCrossentropy(),
      metrics=[tf.keras.metrics.SparseCategoricalAccuracy()])


trainer = tff.learning.build_federated_averaging_process(
    model_fn, client_optimizer_fn=lambda: tf.keras.optimizers.SGD(0.05))


def evaluate(num_rounds=10):
  state = trainer.initialize()
  for round in range(num_rounds):
    t1 = time.time()
    state, metrics = trainer.next(state, train_data)
    t2 = time.time()
    print('Round {}: loss {}, round time {}'.format(round, metrics, t2 - t1))

import grpc

ip_address = '10.0.0.110'  #@param {type:"string"}
port = 8025  #@param {type:"integer"}

client_ex = []
for i in range(40):
  channel = grpc.insecure_channel('{}:{}'.format(ip_address, port))
  client_ex.append(tff.framework.RemoteExecutor(channel, rpc_mode='STREAMING'))

factory = tff.framework.worker_pool_executor_factory(client_ex)
context = tff.framework.ExecutionContext(factory)
tff.framework.set_default_context(context)

evaluate()