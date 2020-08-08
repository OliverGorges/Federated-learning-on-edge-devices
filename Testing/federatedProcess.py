import requests
import json
import time
from multiprocessing import Process, Queue
import os
import numpy
import tensorflow
from utils.Tensorflow.tff import readCheckpointValues
from utils.Tensorflow.trainer import trainer,  augmentData, prepareTFrecord
import json

dataDir = os.path.join("TrainData","data")

def default(obj):
    if type(obj).__module__ == numpy.__name__:
        if isinstance(obj, numpy.ndarray):
            return obj.tolist()
        else:
            return obj.item()
    else:
        return obj.as_list()
    raise TypeError('Unknown type:', type(obj))

def trainWorker(q, id, task):
    print(task)
    subDir = os.path.join(dataDir, f'Subset{id}')
    subResults = os.path.join("TrainData", "output", f'Subset{id}')
    trainer(subResults, subDir, model="ssd_mobilenet_v2_coco_2018_03_29", steps=10)
    print("Train")
    q.put((subResults, task))

def sendData(checkpointDir):
    return f"Success {checkpointDir}"

def run():
    clients = []
    task = []
    startTime = 0
    nr_of_clients = 1


    result = requests.get('http://127.0.0.1:5000/status').json()
    print (result)
    # Submit Task
    if not result['ActiveTask']:
        result = requests.post('http://127.0.0.1:5000/federatedPlan', json=json.loads(json.dumps({"Task": "MaskDetection1", "Data": "Images", "Case": "MaskDetection", "ModelVersion": "TFLite", "MaxClients": 10, "Time": 1594728066})))
        print(result.text)

    # register Clients
    for i in range(nr_of_clients):
        result = requests.get('http://127.0.0.1:5000/reg').json()
        clients.append(result['id'])
        startTime = result['time']

    result = requests.get('http://127.0.0.1:5000/status')
    print (result.json())

    # Wait for Trainphase    
    while time.time() < startTime:
        time.sleep(1)
        print(startTime - time.time())

    result = requests.get('http://127.0.0.1:5000/status')
    print (result.json())

    for c in clients:
        if not c == 0:
            result = requests.get(f'http://127.0.0.1:5000/task/{c}').json()
            print (result)
            task.append(result)

    result = requests.get('http://127.0.0.1:5000/status')
    print (result.json())

    proc = []
    q = Queue()

    for i, t in enumerate(task):
        if t['Accepted']:
            p= Process(target=trainWorker, args=(q, i, t))
            p.start()
            proc.append(p)
            
  
    for p in proc:
        p.join()

    for i in range(q.qsize()):
        res = q.get()



        checkpointDir = res[0]
        task = res[1]
        checkpoint = sorted(os.listdir(checkpointDir)).pop()
        valueDict = readCheckpointValues((os.path.join(checkpointDir, checkpoint), task["Key"]))
        result = requests.post(f'http://127.0.0.1:5000/results/{task["Key"]}', json=json.loads(json.dumps(valueDict, default=default)))
        print(result.text)



if __name__ == "__main__":
    run()