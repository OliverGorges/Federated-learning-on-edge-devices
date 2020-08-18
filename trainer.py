
from utils.Tensorflow.trainer import train_eval,  augmentData, prepareTFrecord
import os
from shutil import copy
import boto3
from botocore.config import Config
from zipfile import ZipFile
import time
import logging 
import requests
import json
from utils.Tensorflow.tff import sendData

logging.propagate = False 

split = 0
save = False
host = "192.168.178.23:5000"#"3.120.138.160:5000"#

def default(obj):
    if type(obj).__module__ == numpy.__name__:
        if isinstance(obj, numpy.ndarray):
            return obj.tolist()
        else:
            return obj.item()
    else:
        return obj.as_list()
    raise TypeError('Unknown type:', type(obj))

def callback(data):
    logging.info(f'http://{host}/meta/{task["Key"]} => {data}')
    requests.post(f'http://{host}/meta/{task["Key"]}', json=json.loads(json.dumps(data, default=default)))


# load Data
mnist_train, mnist_test = tf.keras.datasets.mnist.load_data()

def get_data_for_digit(source, digit):
  output_sequence = []
  all_samples = [i for i, d in enumerate(source[1]) if d == digit]
  for i in range(0, min(len(all_samples), NUM_EXAMPLES_PER_USER), BATCH_SIZE):
    batch_samples = all_samples[i:i + BATCH_SIZE]
    output_sequence.append({
        'x':
            np.array([source[0][i].flatten() / 255.0 for i in batch_samples],
                     dtype=np.float32),
        'y':
            np.array([source[1][i] for i in batch_samples], dtype=np.int32)
    })
  return output_sequence

federated_train_data = [get_data_for_digit(mnist_train, d) for d in range(10)]
federated_test_data = [get_data_for_digit(mnist_test, d) for d in range(10)]
for i in range(10):
    # register Clients
    client = requests.get(f'http://{ host }/reg').json()
    print(client)
    if int(client['id']) == 0:
        exit()


    # Wait for Trainphase    
    logging.info(f"{time.time()}  {client['time']}")
    while time.time() < client['time']:
        time.sleep(5)
        logging.info(client['time'] - time.time())

    task = requests.get(f'http://{ host }/task/{client["id"]}').json()
    logging.info(task)
    # Check if Model exists

    if not task['Accepted']:
        exit()

    taskname = task['Task']

    if 'steps' in task:
        steps = task['steps']
    else:
        steps = 400

    model = tf.keras.models.Sequential([
    tf.keras.layers.Flatten(input_shape=(28, 28, 1)),
    tf.keras.layers.Dense(128,activation='relu'),
    tf.keras.layers.Dense(10, activation='softmax')
    ])
    model.compile(
        loss='sparse_categorical_crossentropy',
        optimizer=tf.keras.optimizers.SGD(0.001),
        metrics=['accuracy'],
    )

    model.fit(
        federated_train_data[i],
        epochs=6,
        validation_data=federated_test_data[1],
    )

"""
    logging.info(f"Traintime on {steps} steps: {time.time()-t1}")
    pipeline = os.path.join(result, "custom_pipeline.config") 
    meta = os.path.join(result, "meta.json")       
    logging.info(f'{result}, {pipeline}, {meta}')
    sendData(f'http://{host}/results/{client["id"]}', result, pipeline, meta)
"""

