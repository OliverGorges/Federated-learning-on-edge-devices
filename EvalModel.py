
from utils.Tensorflow.trainer import eval, augmentData, prepareTFrecord
import os
from shutil import copy, rmtree
import boto3
from botocore.config import Config
from zipfile import ZipFile
import time
import logging

case = "Coco"
annoformat = "JSON"
data = "Images"
subdata = "eval"
model = "MultiClient1"

imgDir = os.path.join("Dataset", case, subdata, data)
annoDir = os.path.join("Dataset", case, subdata, "Annotations")
models = os.path.join("checkpoints", model)
dataDir = os.path.join("TrainData","data")
split = 1
save = False

#Find Labelmap
labelmap = None
files = os.listdir(os.path.join("Dataset", case, subdata))
for f in files:
    if f.startswith("label_map"):
        labelmap = os.path.join("Dataset", case,subdata, f)
        break


tfrecordConfig = prepareTFrecord(imgDir, annoDir, dataDir, labelmap=labelmap, annoFormat=annoformat, split=0.2)
for model in os.listdir(models):
    if model.endswith(".json") or model.startswith("."):
        continue
    modelDir = os.path.join(models, model)
    eval(dataDir, tfRecordsConfig=tfrecordConfig, modelDir= modelDir, steps=1)
