
from utils.Tensorflow.trainer import eval, augmentData, prepareTFrecord
import os
from shutil import copy
import boto3
from botocore.config import Config
from zipfile import ZipFile
import time
import logging 

case = "ThermalFaceDetection"
annoformat = "JSON"
data = "ThermalImages"

model = "ThermalModel40k"

imgDir = os.path.join("Dataset", case, data)
annoDir = os.path.join("Dataset", case, "Annotations")
outDir = os.path.join("TrainData", "model", model)
dataDir = os.path.join("TrainData","data")
split = 1
save = False

#Find Labelmap
labelmap = None
files = os.listdir(os.path.join("Dataset", case))
for f in files:
    if f.startswith("label_map"):
        labelmap = os.path.join("Dataset", case, f)
        break

augImages, augAnnotations = augmentData(imgDir, annoDir, dataDir, split)

tfrecordConfig = prepareTFrecord(augImages[0], augAnnotations[0], dataDir, labelmap=labelmap, annoFormat=annoformat, split=0.7)

eval(outDir, dataDir, tfRecordsConfig=tfrecordConfig, model= model, steps=1)
     