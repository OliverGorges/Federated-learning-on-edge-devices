
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
from utils.Tensorflow.tff import readCheckpointValues

split = 0
save = False
host = "192.168.178.23:5000"

def default(obj):
    if type(obj).__module__ == numpy.__name__:
        if isinstance(obj, numpy.ndarray):
            return obj.tolist()
        else:
            return obj.item()
    else:
        return obj.as_list()
    raise TypeError('Unknown type:', type(obj))

# register Clients
client = requests.get(f'http://{ host }/reg').json()
print(client)
if int(client['id']) == 0:
    exit()

# load Data
compData = os.path.join("Dataset", "data.zip")
result = requests.get(f'http://{ host }/data').json()
print(result["filename"][:-4])
print(os.listdir("Dataset"))
if not result["filename"][:-4] in os.listdir("Dataset"):
    s3 = boto3.client('s3')
    s3.download_file('federatedlearning-cg', f'data/{result["filename"]}', compData)
    with ZipFile(compData, 'r') as zipObj:
        zipObj.extractall("Dataset")

    # remove compressed Data
    os.remove(compData) 



# Wait for Trainphase    
while time.time() < client['time']:
    time.sleep(5)
    logging.info(client['time'] - time.time())

task = requests.get(f'http://{ host }/task/{client["id"]}').json()
logging.info(task)
# Check if Model exists

if not task['Accepted']:
    exit()

taskname = task['Task']
outDir = os.path.join("Traindata", "output", taskname)
data = task['Data']
case = os.listdir("Dataset")[0]
imgDir = os.path.join("Dataset", case, 'images')
annoDir = os.path.join("Dataset", case, "annotations")
# Check Annotation format
if os.listdir(annoDir)[0].endswith('.json'):
    annoformat = "JSON"
elif os.listdir(annoDir)[0].endswith('.xml'):
    annoformat = "XML"

data = task['Data']
dataDir = os.path.join("Traindata","data")

#Find Labelmap
labelmap = None
files = os.listdir(os.path.join("Dataset", case))
for f in files:
    if f.startswith("label_map"):
        labelmap = os.path.join("Dataset", case, f)
        break

if 'steps' in task:
    steps = task['steps']
else:
    steps = 1000

augImages, augAnnotations = augmentData(imgDir, annoDir, dataDir, split)

result = os.path.join("Traindata", "output", taskname)
if not os.path.exists(result):
        os.mkdir(result)
tfrecordConfig = prepareTFrecord(augImages[0], augAnnotations[0], dataDir, labelmap=labelmap, annoFormat=annoformat, split=0.8)
train_eval(result, dataDir, tfRecordsConfig=tfrecordConfig, model=task['ModelVersion'], steps=steps, eval_every_n_steps=200)
        
checkpoint = [f for f in os.listdir(result) if f.endswith('.index')].pop()[:-6]
logging.info(f'Latest Checkpoint: {checkpoint}')


valueDict = readCheckpointValues((os.path.join(result, 'custom_pipeline.config'), os.path.join(result, checkpoint), task["Key"]))
result = requests.post(f'http://{ host }/results/{task["Key"]}', json=json.loads(json.dumps(valueDict, default=default)))
print(result.text)
