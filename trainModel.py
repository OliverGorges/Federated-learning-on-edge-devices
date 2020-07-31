
from utils.Tensorflow.trainer import trainer,  augmentData, prepareTFrecord
import os
from shutil import copy
import boto3
from botocore.config import Config
from zipfile import ZipFile
import time
import logging 

task = "test002"
case = "MaskDetection"
annoformat = "XML"
data = "Images"

imgDir = os.path.join("Dataset", case, data)
annoDir = os.path.join("Dataset", case, "Annotations")
outDir = os.path.join("TrainData", "output", task)
dataDir = os.path.join("TrainData","data")
split = 0
save = False

#Find Labelmap
labelmap = None
files = os.listdir(os.path.join("Dataset", case))
for f in files:
    if f.startswith("label_map"):
        labelmap = os.path.join("Dataset", case, f)
        break

augImages, augAnnotations = augmentData(imgDir, annoDir, dataDir, split)

if split > 1:
    for i in range(split):
        subDir = os.path.join(dataDir, f'Subset{i}')#
        subResults = os.path.join("Traindata", "output", f'Subset{i}')
        copy(os.path.join(dataDir, "labelmap.pbtxt"), os.path.join(subDir, "labelmap.pbtxt"))
        if not os.path.exists(subResults):
            os.mkdir(subResults)
        tfrecordConfig = prepareTFrecord(augImages[i], augAnnotations[i], subDir, labelmap=labelmap, annoFormat=annoformat, split=0.7)

        trainer(subResults, subDir,  tfRecordsConfig=tfrecordConfig, model= "ssd_mobilenet_v2_coco_2018_03_29", steps=10)
else:
    result = os.path.join("Traindata", "output", task)
    if not os.path.exists(result):
            os.mkdir(result)
    tfrecordConfig = prepareTFrecord(augImages[0], augAnnotations[0], dataDir, labelmap=labelmap, annoFormat=annoformat, split=0.7)

    trainer(result, dataDir, tfRecordsConfig=tfrecordConfig, model= "graphmod", steps=10)
        
if save:
    # Upload Results to S3
    checkpoints = sorted(os.listdir(outdir))[-3:]
    zip_file = os.path.join(outDir, 'ckpt.zip')
    with ZipFile(zip_file, 'w') as zip:
        for c in checkpoints:
            zip.write(os.path.join(outDir, c))
        zip.close()
    try:
        s3 = boto3.resources('s3')
        s3.Bucket('federatedlearning').upload_file(zip_file, f'checkpoints/ckpt{time.time()}.zip')
    except:
        logging.info("Cant upload results")
