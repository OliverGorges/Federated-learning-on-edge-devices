from flask import Flask, jsonify, request, send_file, send_from_directory, redirect, render_template
import time
import random
import json
from utils.Tensorflow import tff, trainer
from flask_swagger import swagger
from flask_swagger_ui import get_swaggerui_blueprint
from flask_socketio import SocketIO
from pymongo import MongoClient
from bson.objectid import ObjectId
import logging
import os
from uuid import uuid4
import shutil
import boto3
import zipfile
import tensorflow as tf
from flask_cors import CORS
from multiprocessing import Process
import Demo.demo_run as Demo
import numpy as np

"""
This Server was used for the experiments and simulates multiple trainigsrounds
"""

logging.basicConfig(level=logging.INFO)

app = Flask(__name__,static_folder = "./dist/",
            template_folder = "./dist")
CORS(app)
socketio = SocketIO(app, async_mode="threading")
#client = MongoClient('localhost', 27017)
#db = client.federated_learning

app.config["Models"] = "models"

data = []
workers = 2


timeout = 6000
taskId = "a2adbeda-a565-4180-919f-42da371d377e"
activeTask = 'Test'
aggregated = False
clients = 0
task_path = os.path.join("checkpoints", activeTask)
epoch = 0
cround = 0
result = ""
aggregatedData = {}

def default(obj):
    if type(obj).__module__ == np.__name__:
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        else:
            return obj.item()
    raise TypeError('Unknown type:', type(obj))


@app.route('/reg')
def reg():
    global clients
    clients += 1
    #os.mkdir(os.path.join(task_path, str(clients)))
    return jsonify({'id': clients}), 200

@app.route('/agg_status')
def status():
    return jsonify({'new_checkpoint': aggregated}), 200

@app.route('/workers/<int:w>')
def set_workers(w):
    global workers
    workers = w
    logging.info(f'Workers: {w}')
    return "OK", 200

@app.route('/meta/<int:id>', methods=['POST'])
def setMeta(id):
    """
    Result Upload
    ---
    tags:
        - reporting
        - FederatedProcess
    summary: Endpoint to upload Trainingresults
    consumes:
    - application/json
    produces:
    - application/json
    parameters:
          - in: body
            name: body
            description: dictionary with all trainable parameters
            required: true
    responses:
        200:
            description: Input Accepted
        410:
            description: Invalid input
    """
    global aggregated
    aggregated = False
    trainresult = request.json
    print(trainresult)
    #with open(os.path.join(task_path, str(id), f"meta.json"), 'a') as json_file:
    #    json.dump(trainresult, json_file)
    return "", 200

@app.route('/send_data/<int:id>', methods=['POST'])
def results(id):
    global data
    d = request.json
    print(d)
    data.append(d['data'])
    name = d['name']
    if len(data) >= workers:
        logging.info("Start Aggregation")
        socketio.start_background_task(target=lambda: aggregateThread(name, data))
        data = []
        return "In Progress", 200
    return "OK", 200


@app.route('/agg_result')
def getResults():
    json_result = json.dumps(result, default=default)
    print (len(json_result))
    return jsonify(json_result)

def aggregateThread(names, clients):
    global aggregated
    global epoch
    global result

    print(len(names))
    time.sleep(20)
    new_data = []
    if len(clients) < 1:
        logging.err("No Data")

    for l in range(len(clients[0])):
        layers = [x[l] for x in clients]
        new_layer = []
        for y in range(len(layers[0])):
            input_data = [x[y] for x in layers]
            new_layer.append(np.array(np.average(input_data, axis=0)).astype(np.float32))
        new_data.append(new_layer)
    for i in range(len(new_data)):
        aggregatedData[names[i]] = new_data[i]

    aggregated = True
    epoch += 1
    result = new_data

@app.route('/model')
def getModel():
    return send_from_directory(app.config["Models"],name, as_attachment=True)

@app.route('/creat_model')
def createModel():
    socketio.start_background_task(target=lambda: creatModelThread(aggregatedData))
    return "In Progress", 200

def creatModelThread(aggregatedData):
    global aggregate
    t1 = time.time()
    taskDir = os.path.join("checkpoints", 'Test')
    baseModel = os.path.join( "models", 'tf2_mobilenet')
    newModel = os.path.join( "models", f'Test_{time.time()}')

    t2 = time.time()

    base_checkpoint = os.path.join(baseModel, "checkpoint", "ckpt-0")
    base_pipeline = os.path.join(baseModel, "pipeline.config")
    base = tff.readCheckpointValues((base_pipeline, base_checkpoint , 0))

    if not os.path.exists(newModel):
        os.makedirs(newModel)
    logging.info(aggregatedData.keys())
    tff.writeModel(aggregatedData, base_pipeline, newModel, base_checkpoint, cround)


    #convert TFLITE

    # create zipfile
    files = {}
    files['pipeline.config'] = os.path.join(newModel, "pipeline.config")
    for check in os.listdir(os.path.join(newModel, "checkpoint")):
        files[f'checkpoint/{check}'] = os.path.join(newModel, "checkpoint", check)

    with zipfile.ZipFile(os.path.join(newModel, 'model' +".zip"), 'w', zipfile.ZIP_DEFLATED) as zip:
        for n, f in files.items():
            zip.write(f, n)
        zip.close()

    try:
        s3 = boto3.resources('s3')
        s3.Bucket('federatedlearning-cg').upload_file(os.path.join(newModel, 'model' +".zip"), f'models/{modelVersion}.zip')
    except:
        logging.info("Cant upload model")
    t5 = time.time()

    # Delete Data
    #shutil.rmtree(taskDir)


    def eval_callback(data):
        # update Plan
        logging.info(data)
        results = {}
        for key, metric in data.items():
            if tf.is_tensor(metric):
                results[key] = metric.numpy().tolist()
            else:
                results[key] = metric
        with open(os.path.join(newModel, f"eval.json"), "w") as jsonFile:
            json.dump(results, jsonFile)


    testData = os.path.join('Testdata', 'ThermalDetection')
    if not os.path.exists(testData):
        return
    labelmap = None
    annoFormat = "JSON"
    files = os.listdir(testData)
    for f in files:
        if f.startswith("label_map"):
            labelmap = os.path.join(testData, f)
            if labelmap.endswith("xml"):
                annoFormat = "XML"
            break
    imgDir = os.path.join(testData, 'ThermalImages')
    annoDir = os.path.join(testData, 'Annotations')
    dataDir = os.path.join(testData, 'output')
    if not os.path.exists(dataDir):
        os.mkdir(dataDir)
        os.mkdir(os.path.join(dataDir, "annotations"))
        os.mkdir(os.path.join(dataDir, "images"))

    augImages, augAnnotations = trainer.augmentData(imgDir, annoDir, dataDir, 1)
    tfrecordConfig = trainer.prepareTFrecord(augImages[0], augAnnotations[0], dataDir, labelmap=labelmap, annoFormat=annoFormat, split=0.1)
    trainer.eval(newModel, dataDir, tfRecordsConfig=tfrecordConfig, model= newModel, steps=1, eval_callback=eval_callback)
    shutil.rmtree(dataDir)



app.run(host='0.0.0.0', port=5001, threaded=True)