import requests
import json
import time
from multiprocessing import Process, Queue
import os
import numpy
import tensorflow
import shutil
import zipfile
from utils.Tensorflow.tff import readCheckpointValues, sendData
from utils.Tensorflow.trainer import readEval
import json

dataDir = os.path.join("TrainData","data")
modelDir = os.path.join("TFFSamples")
outputPath = os.path.join("TFFSamples", "results")
host = "10.0.0.110:5000"

def default(obj):
    if type(obj).__module__ == numpy.__name__:
        if isinstance(obj, numpy.ndarray):
            return obj.tolist()
        else:
            return obj.item()
    else:
        return obj.as_list()
    raise TypeError('Unknown type:', type(obj))


def run():
    if os.path.exists(outputPath):
        shutils.rmtree(outputPath)

    results = os.listdir(modelDir)
    clients = []
    task = []
    startTime = 0
    nr_of_clients = len(results)


    result = requests.get(f'http://{ host }/status').json()
    print (result)

    # register Clients
    for i in range(nr_of_clients):
        result = requests.get(f'http://{ host }/reg').json()
        clients.append(result['id'])
        startTime = result['time']

    result = requests.get(f'http://{ host }/status')
    print (result.json())

    # Wait for Trainphase    
    while time.time() < startTime:
        time.sleep(1)
        print(startTime - time.time())

    result = requests.get(f'http://{ host }/status')
    print (result.json())

    for c in clients:
        if not c == 0:
            result = requests.get(f'http://{ host }/task/{c}').json()
            print (result)
            task.append(result)

    result = requests.get(f'http://{ host }/status')
    print (result.json())

    proc = []
    q = Queue()

    for i, t in enumerate(task):
        if t['Accepted']:
            i = int(i)
            checkpointDir = os.path.join(modelDir, results[i])
            pipeline = os.path.join(modelDir, results[i], "custom_pipeline.config")
            meta =  os.path.join(modelDir, results[i],  "meta.json")
            #Eval to json
            data = json.dumps(meta)
            with open(meta, 'w') as outfile:
                json.dump(data, outfile)
            sendData( f'http://{ host }/results/{t["Key"]}', checkpointDir, pipeline, meta)
            

    r = requests.get(f'http://{ host }/aggregate', stream=True)
    with open(os.path.join(modelDir, "result.zip"), 'wb') as fd:
        for chunk in r.iter_content(chunk_size=128):
            fd.write(chunk)

    # Create a ZipFile Object and load sample.zip in it
    with zipfile.ZipFile(os.path.join(modelDir, "result.zip"), 'r') as zipObj:
        # Extract all the contents of zip file in current directory
        zipObj.extractall(outputPath)
    

if __name__ == "__main__":
    run()