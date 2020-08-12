"""
Mathods that are part of the Federated Process
"""

import sys
import tensorflow as tf
import os
import numpy as np
import random
import time
from shutil import copyfile
import json
from object_detection.utils import config_util
from object_detection.builders import model_builder

#Send Data
import socket

from utils.Tensorflow.tfliteConverter import convertModel

# Dummy Model
class MyModel(tf.keras.Model):
    def __init__(self, model):
        super(MyModel, self).__init__()
        self.model = model
        self.seq = tf.keras.Sequential([
            tf.keras.Input([320,320,3], 1),
        ])

    def call(self, x):
        x = self.seq(x)
        images, shapes = self.model.preprocess(x)
        prediction_dict = self.model.predict(images, shapes)
        detections = self.model.postprocess(prediction_dict, shapes)
        boxes = detections['detection_boxes']
        scores = detections['detection_scores'][:,:,None]
        classes = detections['detection_classes'][:,:,None]
        combined = tf.concat([boxes, classes, scores], axis=2)
        return combined

def aggregateVariables(checkpoint_dicts):
    """
    Aggreagtes all values from the checkpoint_dicts to one checkpoint
    """
    chkps = checkpoint_dicts
    aggregatedData = {}
    for x in chkps:
        print('########## Keys ###########')
        #print(list(x.keys()))
    for key in chkps[0].keys():
        if key == "id":
            continue
        if x[key].ndim == 1:
            inputData = [x[key] for x in chkps]
            aggregatedData[key] = np.average(inputData, axis=0, weights=[1 for x in chkps])
        else:
            inputData = [x[key][0] for x in chkps]
            aggregatedData[key] = np.expand_dims(np.average(inputData, axis=0, weights=[1 for x in chkps]), axis=0)

    return aggregatedData

def writeCheckpointValues(data, pipeline, out, checkpoint_file):
    """
    Creates a new checkpoint files based on the new data and the metadata from the ref checkpoint
    data: New Checkpointdata
    ref: reference checkpoint for metadata and keys
    out: output path
    """
    tf.keras.backend.clear_session()

    checkpoint_out = os.path.join(out, "checkpoint", "ckpt")
    pipeline_out = os.path.join(out, "pipeline.config")

    configs = config_util.get_configs_from_pipeline_file(pipeline)
    copyfile(pipeline, pipeline_out)
    detection_model = model_builder.build(configs['model'], is_training=True)

    checkpoint = tf.train.Checkpoint(model=detection_model)
    checkpoint.restore( checkpoint_file).expect_partial()

    dummy = np.random.random([1,320,320,3]).astype(np.float32)
    y = detection_model.predict(tf.convert_to_tensor(dummy, dtype=tf.float32), np.array([320, 320, 3], ndmin=2))

    keys = 0
    failed = 0
    for v in detection_model.trainable_variables:
        if v.name in data:
            print (v.name)
            if data[v.name].shape == v.shape:
                print(v.numpy)
                v.assign(data[v.name])
                print(v.numpy)
            keys += 1
        else:
            #print(v.name)
            failed +=1
    print(f'Keys: {keys}, Failed: {failed}')
    for v in detection_model.trainable_variables:
        if v.name in data:
            print(f'Name: {v.name}')
            print(v.numpy)
            print(data[v.name])
            break
    save = tf.train.Checkpoint(model=detection_model)
    save.save(file_prefix=checkpoint_out)




def readCheckpointValues(path, trainable=True):
    """
    Reads raw data from Checkpoint file
    path: path to checkpoint
    trainable: load just trainable variables, Default: True
    """
    tf.keras.backend.clear_session()
    pipeline = path[0]
    checkpoint = path[1]
    id = path[2]
    values = {}

    randomModi = random.random() + 0.5 # random values between 0.5 and 1.5
    print(f"Load Checkpoint {checkpoint} {id} {randomModi}")
    configs = config_util.get_configs_from_pipeline_file(pipeline)
    detection_model = model_builder.build(configs['model'], is_training=True)

    ckpt = tf.compat.v2.train.Checkpoint(model=detection_model)
    ckpt.restore(checkpoint).expect_partial()
    dummy = np.random.random([1,320,320,3]).astype(np.float32)
    y = detection_model.predict(tf.convert_to_tensor(dummy, dtype=tf.float32), np.array([320, 320, 3], ndmin=2))

    variables = detection_model.trainable_variables
    # Dump 16Bit Json
    for v in variables:
        values[v.name] = np.float16(v.numpy()).tolist() #* randomModi

    data = json.dumps(values)
    with open('data16.json', 'w') as outfile:
        json.dump(data, outfile)

    # Dump 32Bit Json
    for v in variables:
        values[v.name] = v.numpy().tolist() #* randomModi

    data = json.dumps(values)
    with open('data32.json', 'w') as outfile:
        json.dump(data, outfile)

    """
    names = [weight.name for layer in model.layers for weight in layer.weights]
    weights = model.get_weights()

    for name, weight in zip(names, weights):
        if name.startswith('my_model_'):
            name = "my_model" + name[10:]
        values[name] = weight
    """
    print(f"{len(values)} restored")
    #for var in variables:
        #values[var[0]] = ckpt.get_tensor(var[0]).numpy()


    return values


def sendData(path, host, port, trainable=True):
    """
    Sends raw checkpoint data to another device/server
    """
    readCheckpointValues(path, trainable=trainable)

    #Init Socket
    socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.connect((host, port))

    pass

if __name__ == "__main__":
    readCheckpointValues(("Traindata\\output\\test002\\custom_pipeline.config","Traindata\\output\\test002\\ckpt-3", 1))
    exit()
    model_id = "FederatedTestModel"
    modelDir = os.path.join("Traindata", "model", model_id)
    pipeline = os.path.join(modelDir, "custom_pipeline.config")
    checkpoints = []
    checkpoint = os.path.join(modelDir, "checkpoint", "ckpt-43")
    nrbs = [42 ,43]
    for nr in nrbs:
        checkpoints.append(os.path.join(modelDir, "checkpoint", f"ckpt-{nr}"))


    output = os.path.join(modelDir, 'tflite')
    convertModel(modelDir, output)

    print(checkpoints)
    #stage1 = mpipe.UnorderedStage(readCheckpointValues, len(checkpoints))
    #pipe = mpipe.Pipeline(stage1)
    chkps = []
    aggregatedData = {}
    t1 = time.time()
    #chkps.append(readCheckpointValues(checkpoints[0], "Ref", False)) # load Reference Checkpoint with all Keys
    t2 = time.time()
    print(f'Load Ref: {t2-t1}')
    for i, c in enumerate(checkpoints):
        chkps.append(readCheckpointValues((pipeline, c , i)))

    t3 = time.time()
    print(f'Load Ref: {t2-t1}, Load Checkpoints: {t3-t2}')
    aggregatedData = aggregateVariables(chkps)
    t4 = time.time()
    print(f'Load Ref: {t2-t1}, Load Checkpoints: {t3-t2}, Modify Key: {t4-t3}')

    writeCheckpointValues(aggregatedData, pipeline, os.path.join("Traindata", "model", "graphmod" ), checkpoint)
    t5 = time.time()
    print(f'Load Ref: {t2-t1}, Load Checkpoints: {t3-t2}, Modify Key: {t4-t3}, Save Checkpoint: {t5-t4}')
    output = os.path.join("Traindata", "model", "graphmod", "tflite" )
    convertModel(os.path.join("Traindata", "model", "graphmod") , output)
