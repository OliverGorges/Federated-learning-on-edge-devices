
import os
from zipfile import ZipFile
import time
import logging
import requests
import json
import random
import zipfile
import multiprocessing

host = "0.0.0.0:5000"

def zipCheckpoint(srcDict, dst):
    zf = zipfile.ZipFile(dst, "w", zipfile.ZIP_DEFLATED)
    for filename in srcDict.keys():

        zf.write(srcDict[filename], filename)
    zf.close()

def sendData( url, folder, checkpointDir, pipeline, meta):
    """
    Sends raw checkpoint data to another device/server
    """
    files = {}
    files['pipeline.config'] = pipeline
    files['meta.json'] = meta
    latest_checkpoint = "ckpt-5"
    print(latest_checkpoint)
    for f in os.listdir(checkpointDir):
        if f.startswith(latest_checkpoint):
            files[f'checkpoint/{f}'] = os.path.join(checkpointDir, f)
    print(files)
    out = os.path.join(folder, "temp.zip")
    zipCheckpoint(files, out)
    fileobj = open(out, 'rb')
    print(fileobj)
    r = requests.post(url, data = {"mysubmit":"Go"}, files={"file": ("test.zip", fileobj)})



def client(id):

    logging.info(f"Start CLient {id}")
    data_id = random.randint(1, 5)
    delay = random.randint(3, 10)
    time.sleep( random.randint(20, 60))
    # register Clients
    client = requests.get(f'http://{ host }/reg').json()
    logging.info(client)
    if int(client['id']) == 0:
        return


    # Wait for Trainphase
    logging.info(f"{time.time()}  {client['time']}")
    while time.time() < client['time']:
        time.sleep(5)
        logging.info(client['time'] - time.time())


    plan = requests.get(f'http://{ host }/task/{client["id"]}').json()
    logging.info(plan)
    # Check if Model exists

    task = plan['Task']
    case = plan['Case']
    data = plan['Data']
    model = plan['ModelVersion']
    steps = 21

    for s in range(steps):
        requests.post(f'http://{host}/meta/{plan["Key"]}', json=json.loads(json.dumps({'step': s*5, 'loss': random.random(), 'acc': 0.5+random.random()})))
        time.sleep(delay)

    fileobj = open(os.path.join("Demo", f"client_{data_id}.zip"), 'rb')
    print(fileobj)
    r = requests.post(f'http://{host}/results/{client["id"]}', data = {"mysubmit":"Go"}, files={"file": ("test.zip", fileobj)})

    #result = os.path.join("Demo", f"client_{id}", "checkpoint")
    #pipeline =  os.path.join("Demo", f"client_{id}", "pipeline.config")
    #evalData =  os.path.join("Demo", f"client_{id}", "eval.json")
    #sendData(f'http://{host}/results/{client["id"]}',os.path.join("Demo", f"client_{id}"),  result, pipeline, evalData)



"""
for id in range(10):
    p = Process(target=client, args=(id, ))
    p.start()
    time.sleep(10)

"""

def run(clients):
    p = multiprocessing.Pool()
    p.map(client, range(clients))

if __name__ == "__main__":
    run(10)
