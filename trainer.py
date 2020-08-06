
from utils.Tensorflow.trainer import trainer,  augmentData, prepareTFrecord
import os
from shutil import copy
import boto3
from botocore.config import Config
from zipfile import ZipFile
import time
import logging 



split = 0
save = False

# register Clients
client = requests.get('http://127.0.0.1:5000/reg').json()

# load Data
compData = os.path.join("Dataset", "data.zip")
result = requests.get('http://127.0.0.1:5000/data').json()

s3 = boto3.client('s3')
s3.download_file('federatedlearning-cg', f'data/{result['filename']}', compData)
with ZipFile(compData, 'r') as zipObj:
   zipObj.extractall()

# remove compressed Data
os.remove(compData) 

# Wait for Trainphase    
while time.time() < client['time']:
    time.sleep(5)
    logging.info(client['time'] - time.time())

task = requests.get(f'http://127.0.0.1:5000/task/{client['id']}').json()

# Check if Model exists


taskname = task['Task']
outDir = os.path.join("Traindata", "output", taskname)

case = os.listdir("Dataset")[0]
imgDir = os.path.join("Dataset", case, data)
annoDir = os.path.join("Dataset", case, "Annotations")
# Check Annotation format
if os.listdir(annoDir)[0].endswith('.json')
    annoformat = "JSON"
elif if os.listdir[annoDir)[0].endswith('.xml')
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


augImages, augAnnotations = augmentData(imgDir, annoDir, dataDir, split)

result = os.path.join("Traindata", "output", task)
if not os.path.exists(result):
        os.mkdir(result)
tfrecordConfig = prepareTFrecord(augImages[0], augAnnotations[0], dataDir, labelmap=labelmap, annoFormat=annoformat, split=0.8)
trainer(result, dataDir, tfRecordsConfig=tfrecordConfig, model=task['model'], steps=task['steps'])
        
checkpoint = [f for f in os.listdir(result) if f.endswith('.index')].pop()[:-6]
logging.info(f'Latest Checkpoint: {checkpoint}')


valueDict = readCheckpointValues((os.path.join(result, 'custom_pipeline.config'), os.path.join(result, checkpoint), task["Key"]))
result = requests.post(f'http://127.0.0.1:5000/results/{task["Key"]}', json=json.loads(json.dumps(valueDict, default=default)))
print(result.text)
