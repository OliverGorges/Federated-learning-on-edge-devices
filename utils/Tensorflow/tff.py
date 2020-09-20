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
import logging
from object_detection.utils import config_util
from object_detection.builders import model_builder

from utils.Tensorflow.tfliteConverter import convertModel



# Dummy Model
class MyModel(tf.keras.Model):
    def __init__(self, model):
        super(MyModel, self).__init__()
        self.model = model
        self.seq = tf.keras.Sequential([
            tf.keras.Input([300,300,3], 1),
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


def delta(base, checkpoint):
    delta = {}
    for key in checkpoint.keys():
        delta[key] = base[key] - checkpoint[key]

    sample = random.choice(list(checkpoint.keys()))
    #logging.debug(f'Key: {sample}, Base: {base[sample]}, Checkpoint: {checkpoint[sample]}, Delta: {delta[sample]}')
    return delta

def aggregateVariables(checkpoint_dicts):
    """
    Aggreagtes all values from the checkpoint_dicts to one checkpoint
    """
    chkps = checkpoint_dicts
    aggregatedData = {}
    print(chkps[0].keys())
    for key in chkps[0].keys():
        print(key)
        if "extractor" not in key:
            continue

        if np.array(chkps[0][key]).ndim == 1:
            inputData = [x[key] for x in chkps]
            aggregatedData[key] = np.average(inputData, axis=0, weights=[1 for x in chkps])
        else:
            inputData = [x[key][0] for x in chkps]
            aggregatedData[key] = np.expand_dims(np.average(inputData, axis=0, weights=[1 for x in chkps]), axis=0)

    return aggregatedData

def writeModel(aggregatedDelta, base_pipeline, out, base_checkpoint, c_round=0):
    tf.keras.backend.clear_session()


    #base_checkpoint = os.path.join(baseModel, "checkpoint", "ckpt-1")
    #base_pipeline = os.path.join(baseModel, "pipeline.config")

    checkpoint_out = os.path.join(out, "checkpoint", f"ckpt-{c_round}")
    pipeline_out = os.path.join(out, "pipeline.config")

    configs = config_util.get_configs_from_pipeline_file(base_pipeline)
    copyfile(base_pipeline, pipeline_out)
    detection_model = model_builder.build(configs['model'], is_training=True)

    checkpoint = tf.train.Checkpoint(model=detection_model)
    checkpoint.restore(base_checkpoint).expect_partial()

    dummy = np.random.random([1,300,300,3]).astype(np.float32)
    y = detection_model.predict(tf.convert_to_tensor(dummy, dtype=tf.float32), np.array([300, 300, 3], ndmin=2))

    keys = 0
    failed = 0
    logging.debug(aggregatedDelta.keys())

    backbone_model = detection_model._feature_extractor.classification_backbone
    logging.debug(backbone_model.layers)
    for layer in backbone_model.layers:
        if layer.name in aggregatedDelta:
            print(aggregatedDelta[layer.name])
            print(layer.get_weights())
            layer.set_weights(np.array(aggregatedDelta[layer.name]))
    """
    logging.debug([v.name for v in detection_model.trainable_variables])
    for v in detection_model.trainable_variables:
        if v.name in aggregatedDelta:
            if aggregatedDelta[v.name].shape == v.shape:
                v.assign(v.numpy() + aggregatedDelta[v.name])
            keys += 1
        else:
            #logging.debug(v.name)
            failed +=1
    logging.debug(f'Keys: {keys}, Failed: {failed}')
"""
    save = tf.train.Checkpoint(model=detection_model)
    save.save(file_prefix=checkpoint_out)

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

    dummy = np.random.random([1,300,300,3]).astype(np.float32)
    y = detection_model.predict(tf.convert_to_tensor(dummy, dtype=tf.float32), np.array([300, 300, 3], ndmin=2))

    keys = 0
    failed = 0
    for v in detection_model.trainable_variables:
        if v.name in data:
            if data[v.name].shape == v.shape:
                v.assign(data[v.name])
            keys += 1
        else:
            #logging.debug(v.name)
            failed +=1
    logging.debug(f'Keys: {keys}, Failed: {failed}')

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
    logging.debug(f"Load Checkpoint {checkpoint} {id} {randomModi}")
    configs = config_util.get_configs_from_pipeline_file(pipeline)
    detection_model = model_builder.build(configs['model'], is_training=True)

    ckpt = tf.compat.v2.train.Checkpoint(model=detection_model)
    ckpt.restore(checkpoint).expect_partial()
    dummy = np.random.random([1,300,300,3]).astype(np.float32)
    y = detection_model.predict(tf.convert_to_tensor(dummy, dtype=tf.float32), np.array([300, 300, 3], ndmin=2))

    variables = detection_model.trainable_variables

    for v in variables:
        if "extractor" not in v.name:
            continue
        values[v.name] = v.numpy()#.tolist() #* randomModic

    """
    names = [weight.name for layer in model.layers for weight in layer.weights]
    weights = model.get_weights()

    for name, weight in zip(names, weights):
        if name.startswith('my_model_'):
            name = "my_model" + name[10:]
        values[name] = weight
    """
    logging.debug(f"{len(values)} restored")
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
    #readCheckpointValues(("Traindata\\output\\test002\\custom_pipeline.config","Traindata\\output\\test002\\ckpt-3", 1))
    #Log
    logging.basicConfig(level=logging.DEBUG)
    baseModel = os.path.join("models", "Animals_1597989179")
    base_checkpoint = os.path.join(baseModel, "checkpoint", "ckpt-1")
    base_pipeline = os.path.join(baseModel, "pipeline.config")

    model_id = "zebra_ckpt1597965633.0881982"
    modelDir = os.path.join("checkpoints", "Animals", model_id)
    pipeline = os.path.join(modelDir, "pipeline.config")
    checkpoints = []
    checkpoint = os.path.join(modelDir, "checkpoint", "ckpt-7")
    nrbs = [7 ,7, 7]
    for nr in nrbs:
        checkpoints.append(os.path.join(modelDir, "checkpoint", f"ckpt-{nr}"))

    logging.debug(checkpoints)
    #stage1 = mpipe.UnorderedStage(readCheckpointValues, len(checkpoints))
    #pipe = mpipe.Pipeline(stage1)
    chkps = []
    aggregatedData = {}
    t1 = time.time()

    base = readCheckpointValues((base_pipeline, base_checkpoint , 0))
    t2 = time.time()
    logging.debug(f'Load Ref: {t2-t1}')

    for i, c in enumerate(checkpoints):
        check = readCheckpointValues((pipeline, c , i))
        chkps.append(delta(base, check))
    t3 = time.time()
    logging.debug(f'Load Ref: {t2-t1}, Load Checkpoints: {t3-t2}')

    aggregatedData = aggregateVariables(chkps)
    t4 = time.time()
    logging.debug(f'Load Ref: {t2-t1}, Load Checkpoints: {t3-t2}, Modify Key: {t4-t3}')

    writeModel( aggregatedData, baseModel, os.path.join("models", "AnimalModelNew"))
    t5 = time.time()
    logging.debug(f'Load Ref: {t2-t1}, Load Checkpoints: {t3-t2}, Modify Key: {t4-t3}, Save Checkpoint: {t5-t4}')

