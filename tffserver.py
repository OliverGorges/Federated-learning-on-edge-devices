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

logging.basicConfig(level=logging.INFO)

app = Flask(__name__,static_folder = "./dist/",
            template_folder = "./dist")
CORS(app)
socketio = SocketIO(app, async_mode="threading")
#client = MongoClient('localhost', 27017)
#db = client.federated_learning

app.config["Models"] = "models"

data = {}

SWAGGER_URL =  '/docs'
swaggerui_blueprint = get_swaggerui_blueprint(
    SWAGGER_URL,
    '/spec',
    config={
        'app_name': "Federated Server"
    },
)

timeout = 6000
taskId = "a2adbeda-a565-4180-919f-42da371d377e"
activeTask = None
aggregate = False
nextTime = 0
clients = {}
registeredClients = 0
validKeys = []
completedTasks = 0
Step = 0

def changeStep(newStep):
    global Step
    Step = newStep
    socketio.emit("status", {"Step": Step, "nextTime": nextTime, "endTime": 0 if nextTime==0 else nextTime+timeout,  "ActiveTask": False if taskId==0 else True ,"RegisteredClients": registeredClients, "AcceptedClients": len(clients.keys())+completedTasks, "ActiveTasks": len(validKeys), "CompletedTasks": completedTasks }, broadcast=True)

def updateStatus():
    socketio.emit("status", {"Step": Step, "nextTime": nextTime, "endTime": 0 if nextTime==0 else nextTime+timeout,  "ActiveTask": False if taskId==0 else True ,"RegisteredClients": registeredClients, "AcceptedClients": len(clients.keys())+completedTasks, "ActiveTasks": len(validKeys), "CompletedTasks": completedTasks }, broadcast=True)

@socketio.on('connect')
def handle_connect():
    global Step
    print('Client connected')
    if Step == 0:
        Step = 1
    socketio.emit("status", {"Step": Step, "nextTime": nextTime, "endTime": 0 if nextTime==0 else nextTime+timeout,  "ActiveTask": False if taskId==0 else True ,"RegisteredClients": registeredClients, "AcceptedClients": len(clients.keys())+len(validKeys)+completedTasks, "ActiveTasks": len(validKeys), "CompletedTasks": completedTasks })


@app.route('/', defaults={'path': ''})
@app.route('/<path:path>')
def catch_all(path):
    return render_template("index.html")


@app.route('/data')
def dataset():
    """
    Phase Status
    ---
    summary: returns current dataset for virtual clients
    tags:
        - demo
    produces:
    - application/json
    """
    return jsonify({ "filename": "client_1.zip" }), 202

# Federaded Process

@app.route('/reg')
def register():
    """
    Register available Client
    ---
    tags:
        - selection
        - FederatedProcess
    responses:
        202:
            description: Client Accepted
            content:
                application/json:
                    schema:
                        type: object
                        properties:
                            time:
                                type: string
                            id:
                                type: string
        205:
            description: Client postponed
            content:
                application/json:
                    schema:
                        type: object
                        properties:
                            time:
                                type: string
                            id:
                                type: string
    """
    global registeredClients
    global clients
    registeredClients += 1
    plan = json.load(open(os.path.join("tasks", f"{taskId}.json")))
    # When client is out of timeframe or selected by dropout
    if  random.random() > 0.8 or len(clients) >= int(plan["MaxClients"]):
        return jsonify({ "time": 0, "id": 0 }), 205
    id = random.randint(1000, 9999)
    clients[id] = {'step': 0, 'loss': -1, 'acc': -1}
    updateStatus()
    return jsonify({ "time": nextTime, "id": id }), 202

@app.route('/task/<int:client_id>')
def getTask(client_id):
    """
    Get new TFFPlan
    ---
    summary: Enpoint to get the Trainingsplan for this Phase
    tags:
        - configuration
        - FederatedProcess
    produces:
    - application/json
    parameters:
      - in: path
        name: path
        description: id from the registration that allows the client to participate in this phase
        required: true
    responses:
        200:
            content:
                application/json:
                    schema:
                        type: object
                        properties:
                            Accepted:
                                type: boolean
                            Task:
                                type: string
                            Data:
                                type: string
                            Case:
                                type: string
                            ModelVersion:
                                type: string
        404:
            description: Cant find FederatedPlan
        410:
            description: Request not in Timeslot
    """
    global clients
    global validKeys
    global activeTask
    # Last possible time is 30 min before Timeout
    if time.time() <= nextTime+timeout:
        accepted = False
        #plan = db.plans.find_one({"_id": ObjectId(str(taskId))})
        plan = json.load(open(os.path.join("tasks", f"{taskId}.json")))
        if plan is None:
            logging.info(f"cant find Task {taskId}")
            return "cant find Task", 404
        activeTask = plan['Task']

        if client_id in clients.keys():
            accepted = True
            validKeys.append(client_id)
            os.makedirs(os.path.join("checkpoints", activeTask, str(client_id)))
        if accepted:
            updateStatus()
            return jsonify({ "Accepted": accepted, "Key": client_id, "Task": plan['Task'], "Data": plan['Data'], "Case": plan['Case'], "ModelVersion": plan['ModelVersion'] }), 200
        elif client_id == 0:
            #Create folder for client results
            return jsonify({ "Accepted": accepted, "Key": client_id, "Task": plan['Task'], "Data": plan['Data'], "Case": plan['Case'], "ModelVersion": plan['ModelVersion'] }), 200
    else:
        changeStep(0)
    logging.info(f'{time.time()} <= {nextTime+timeout}')
    return jsonify({ "Accepted": False }), 410


@app.route('/model/<name>')
def updateModel(name):
    """
    Get new TFFPlan
    ---
    summary: Enpoint to get the Trainingsplan for this Phase
    tags:
        - configuration
        - FederatedProcess
    produces:
    - application/json
    parameters:
      - in: path
        name: path
        description: id from the registration that allows the client to participate in this phase
        required: true
    responses:
        200:
            description: Zip file with Model Checkpoint

    """

    modelDir = os.path.join("models", name)
    zip_file = os.path.join(modelDir, f'{name}.zip')
    if not os.path.exists(zip_file):
        # create zipfile
        files = {}
        files['pipeline.config'] = os.path.join(modelDir, "pipeline.config")
        for check in os.listdir(os.path.join(modelDir, "checkpoints")):
            files[f'checkpoint/{check}'] = os.path.join(modelDir, "checkpoints", check)

        with zipfile.ZipFile(zip_file, 'w', zipfile.ZIP_DEFLATED) as zip:
            for name, f in files.items():
                zip.write(f, name)
            zip.close()
    return send_file(zip_file, attachment_filename='checkpoints.zip', as_attachment=True), 200

@app.route('/results/<int:id>', methods=['POST'])
def results(id):
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
    global data
    global completedTasks
    global validKeys

    if nextTime <= time.time() <= nextTime+timeout and not aggregate and activeTask:
        if id in validKeys :

            # check if the post request has the file part
            if not 'file' in request.files:
                return "", 404
            file = request.files['file']
            # if user does not select file, browser also
            # submit an empty part without filename
            if file.filename == '':
                flash('No selected file')
                return "", 404
            if file:
                del clients[id]
                validKeys.remove(id)
                completedTasks = completedTasks + 1
                updateStatus()
                try:
                    zip_file = os.path.join("checkpoints", activeTask, str(id), 'results.zip')
                    file.save(zip_file)
                    with zipfile.ZipFile(zip_file, 'r') as zip_ref:
                        zip_ref.extractall(os.path.join("checkpoints", activeTask, str(id)))
                    os.remove(zip_file)
                    return "", 200
                except:
                    return "", 500

        return "", 400
    else:
        return "", 410

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
    if not aggregate and activeTask:
        changeStep(3)
        trainresult = request.json
        print(f'JSON: {trainresult}')
        clients[id] = trainresult
    #with open(os.path.join("checkpoints", activeTask, str(id), f"meta.json"), 'w') as json_file:
    #        json.dump(trainresult, json_file)
    return "", 200

@app.route('/meta')
def getMeta():
    data = []
    logging.info(clients)
    for id, client in enumerate(clients.values()):
        client['id'] = id
        data.append(client)
    return jsonify(data), 200

# Internal endpoints


@app.route('/federatedPlan', methods=['POST', 'PUT'])
def uploadPlan():
    """
    Endpoint to upload the Federatedplan for the next Phase
    ---
    tags:
        - internal
    summary: Upload FederatedPlan
    description: Internal Endpoint for Server/Training configuration
    consumes:
    - application/json
    responses:
        200:
            description: Input Accepted
        410:
            description: Invalid input
    """
    global taskId
    global nextTime
    global clients
    global registeredClients
    global validKeys
    global completedTasks
    global aggregate

    clients = {}
    registeredClients = 0
    validKeys = []
    completedTasks = 0
    aggregate = False

    if request.form:
        plan = request.form.to_dict()
    else:
        plan = request.json

    if request.method == 'POST':
        keys = ["Task", "Data", "Case", "ModelVersion", "Time", "MaxClients"]
        for key in keys:
            if not key in plan:
                return f"Plan must incluse the key: {key}", 410
        utime = int(plan['Time'])/1000
        if utime  < time.time():
            logging.info("Timeerror")
            #return "Time not valid", 410
            utime = time.time()+20

        # Create new Task
        plan['Task'] = f"{plan['ModelVersion']}_{plan['Task']}"

        #dbId = db.plans.insert_one(plan).inserted_id
        taskId = f"{plan['Task']}_{time.time()}"
        output = os.path.join("checkpoints", plan['Task'])
        if os.path.exists(output):
            shutil.rmtree(output)
        os.makedirs(output)

        with open(os.path.join(output, f"definition.json"), 'w') as json_file:
            json.dump(plan, json_file)

        with open(os.path.join("tasks", f"{taskId}.json"), 'w') as json_file:
            json.dump(plan, json_file)
        nextTime = utime
        changeStep(2)
        return redirect(request.referrer, code=302)
    else:
        # Update/Rerun Plan
        if not "ID" in plan:
            return f"Plan must incluse the key: ID", 410
        if not "Time" in plan:
            return f"Plan must incluse the key: Time", 410
        utime = int(plan['Time'])
        if utime < time.time():
            logging.info("Timeerror")
            #return "Time not valid", 410
            utime = time.time()+120

        oldPlan = json.load(open(os.path.join("tasks", f"{Task['_id']}.json")))
        oldPlan.pop("_id", None)

        for k in plan.keys():
            oldPlan[k] = plan[k]
        plan['Task'] = plan['Task'] + "1"
        taskId = f"{plan['Task']}_{time.time()}"
        with open(os.path.join("tasks", f"{taskId}.json"), 'w') as json_file:
            json.dump(plan, json_file)
        nextTime = plan['Time']

        changeStep(2)
        return redirect(request.referrer, code=302)



@app.route('/aggregate')
def aggregateResults():
    """
    Endpoint that aggregates all Checkpoints from this TrainingsPhase
    ---
    tags:
        - internal
        - FederatedProcess
    summary: Endpoint to aggregate Trainingresults
    description: Internal Endpoint that gets triggered after the timeout to aggregate all incomming Training result to a new Checkpointfile
    produces:
    - CheckpointFile
    responses:
        200:
            description: Input Accepted
    """
    global aggregate
    aggregate  = True
    changeStep(4)

    plan = json.load(open(os.path.join("tasks", f"{taskId}.json")))

    results = os.listdir(os.path.join("checkpoints", plan['Task']))
    if len(results) <=1:
        return "not ready", 500

    logging.info(results)

    socketio.emit('aggregation', {"status": 0, "msg": f"Start aggregation with {len(results)-1} submits"}, broadcast=True)
    socketio.start_background_task(target=lambda: aggregateThread(plan, taskId, results))

    #p = Process(target=aggregateThread, args=(plan, taskId, results, socketio))
    #p.start()
    return "In Progress", 200

def aggregateThread(plan, taskId, clients):
    global aggregate
    t1 = time.time()
    taskDir = os.path.join("checkpoints", plan['Task'])
    baseModel = os.path.join( "models", plan['ModelVersion'])
    newModel = os.path.join( "models", plan['Task'])

    t2 = time.time()

    base_checkpoint = os.path.join(baseModel, "checkpoint", "ckpt-0")
    base_pipeline = os.path.join(baseModel, "pipeline.config")
    base = tff.readCheckpointValues((base_pipeline, base_checkpoint , 0))
    checkpoint = ""
    chkps = []
    for i, result in enumerate(clients):
        if result.endswith('.json') or result.startswith("."):
            continue
        result = os.path.join(taskDir, result)
        try:
            checkpointDir = os.path.join( result, "checkpoint")
            for f in os.listdir(checkpointDir):
                if f.endswith('.index'):
                    c = os.path.join(checkpointDir, f[:-6])
                    break
            if c == None:
                continue
            pipeline = os.path.join(result, 'pipeline.config')

            checkpoint_data = tff.readCheckpointValues((pipeline, c , i))
            chkps.append(tff.delta(base, checkpoint_data))
        except:
            logging.warning(f"{result} failed")
            continue
        socketio.emit('aggregation', {"status": 0, "msg": f"{i}/{len(clients)-1} checkpoints processed"}, broadcast=True)
        checkpoint = c
    t3 = time.time()
    logging.info(f'Load Ref: {t2-t1}, Load Checkpoints: {t3-t2}')
    aggregatedData = tff.aggregateVariables(chkps)
    socketio.emit('aggregation', {"status": 0, "msg": f"New Model aggragated"}, broadcast=True)
    t4 = time.time()
    logging.info(f'Load Ref: {t2-t1}, Load Checkpoints: {t3-t2}, Modify Key: {t4-t3}')

    if not os.path.exists(newModel):
        os.makedirs(newModel)


    plan['newModel'] = plan['Task']

    # update Plan
    with open(os.path.join("tasks", f"{taskId}.json"), "w") as jsonFile:
        json.dump(plan, jsonFile)

    tff.writeModel(aggregatedData, base_pipeline, newModel, base_checkpoint)


    #convert TFLITE

    # create zipfile
    files = {}
    files['pipeline.config'] = os.path.join(newModel, "pipeline.config")
    for check in os.listdir(os.path.join(newModel, "checkpoint")):
        files[f'checkpoint/{check}'] = os.path.join(newModel, "checkpoint", check)

    with zipfile.ZipFile(os.path.join(newModel, plan['Task'] +".zip"), 'w', zipfile.ZIP_DEFLATED) as zip:
        for name, f in files.items():
            zip.write(f, name)
        zip.close()

    try:
        s3 = boto3.resources('s3')
        s3.Bucket('federatedlearning-cg').upload_file(os.path.join(newModel, plan['Task'] +".zip"), f'models/{modelVersion}.zip')
        socketio.emit('aggregation', {"status": 1, "msg": "Model saved"}, broadcast=True)
    except:
        logging.info("Cant upload model")
        socketio.emit('aggregation', {"status": 1, "msg": "Model saved localy"}, broadcast=True)
    t5 = time.time()
    logging.info(f'Load Ref: {t2-t1}, Load Checkpoints: {t3-t2}, Modify Key: {t4-t3}, Save Checkpoint: {t5-t4}')

    # Delete Data
    shutil.rmtree(taskDir)


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

        socketio.emit('aggregation', {"status": 3, "msg": "Evaluation done"}, broadcast=True)

    testData = os.path.join('Testdata', plan['Case'])
    if not os.path.exists(testData):
        socketio.emit('aggregation', {"status": 0, "msg": "Evaluation Failed: No Evaldata"}, broadcast=True)
        changeStep(1)
        return
    socketio.emit('aggregation', {"status": 2, "msg": "Start Evaluation"}, broadcast=True)
    labelmap = None
    annoFormat = "JSON"
    files = os.listdir(testData)
    for f in files:
        if f.startswith("label_map"):
            labelmap = os.path.join(testData, f)
            if labelmap.endswith("xml"):
                annoFormat = "XML"
            break
    imgDir = os.path.join(testData, 'Images')
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
    changeStep(1)


@app.route('/eval')
def evalResults():
    """
    Phase Status
    ---
    summary: Eval Model Results
    tags:
        - reporting
    produces:
    - application/json
    """

    plan = json.load(open(os.path.join("tasks", f"{taskId}.json")))

    old = json.load(open(os.path.join("models", plan['ModelVersion'], 'eval.json')))
    new = json.load(open(os.path.join("models", plan['newModel'], 'eval.json')))

    results = []
    for key in new.keys():
        try:
            results.append({'metric': key.split('/').pop(), 'new': new[key], 'old': old[key]})
        except:
            logging.warning(f"Error with Key {key}")

    return jsonify([old, new])


@app.route('/wipeall')
def wipeData():
    """
    Phase Status
    ---
    summary: Wipes all produces Data
    tags:
        - demo
    """

    global taskId
    global validKeys
    global activeTask
    global nextTime
    global registeredClients
    global clients

    shutil.rmtree('tasks/')
    os.makedirs('tasks/')
    shutil.rmtree('checkpoints/')
    os.makedirs('checkpoints/')
    for model in os.listdir('models'):
        if model == "tf2_mobilenet":
            continue
        shutil.rmtree(os.path.join('models', model))

    activeTask = False
    clients = {}
    nextTime = 0
    validKeys = []
    taskId == 0
    registeredClients = 0
    changeStep(0)
    return 200

@app.route('/cases')
def listCases():
    """
    Phase Status
    ---
    summary: Lists available eval data for cases
    produces:
    - application/json
    """
    cases = []
    for c in os.listdir("Testdata"):
        if c.startswith('.'):
            continue
        data = []
        label = {}
        for d in os.listdir(os.path.join("Testdata",c)):
            if d.startswith('.'):
                continue
            if not d.startswith('label_map') and not d.startswith('Annotations') and not d.startswith('output'):
                data.append(d)
            if d == 'label_map.json':
                label = json.load(open(os.path.join("Testdata", c, f"label_map.json")))
        case = {'name': c, 'data': data}
        case.update(label)
        cases.append(case)
    return jsonify(cases)

@app.route('/models')
def listModels():
    """
    Phase Status
    ---
    summary: Lists all models on Server
    tags:
        - internal
    produces:
    - application/json
    """
    models = []
    if Step == 0:
        changeStep(1)
    for model in os.listdir("models"):
        if not model.startswith('.'):
            try:
                eval = json.load(open(os.path.join("models", model, f"eval.json")))
                logging.info(eval)
                mAP = eval["DetectionBoxes_Precision/mAP"]
                ar = eval["DetectionBoxes_Recall/AR@10"]
                loss = eval["Loss/total_loss"]
            except:
                mAP = 0
                ar = 0
                loss = 0
            models.append({'name': model, 'mAP': mAP, 'ar': ar, 'loss': loss})
    return jsonify(models)



@app.route('/model/<name>')
def sendModel(name):
    """
    Phase Status
    ---
    summary: Sends modelfile to client
    tags:
        - internal
    produces:
    - application/zip
    """
    return send_from_directory(app.config["Models"],name, as_attachment=True)



## Info Endpoints
@app.route('/status')
def getStatus():
    """
    Phase Status
    ---
    summary: Endpoint that returns the Status of the current Phase
    description: Returns Status informations for a optinal UI
    tags:
        - internal
    produces:
    - application/json
    responses:
        200:
            description: Json with Status Information
            content:
                application/json:
                    schema:
                        type: object
                        properties:
                            RegisteredClients:
                                type: int
                            AcceptedClients:
                                type: int
                            ActiveTasks:
                                type: int
                            CompletedTasks:
                                type: int
    """


    if time.time() > nextTime+timeout:
        resetTask()

    return jsonify({"Step": Step, "nextTime": nextTime, "endTime": 0 if nextTime==0 else nextTime+timeout,  "ActiveTask": False if taskId==0 else True ,"RegisteredClients": registeredClients, "AcceptedClients": len(clients.keys())+completedTasks, "ActiveTasks": len(validKeys), "CompletedTasks": completedTasks })

def resetTask():
    global taskId
    global validKeys
    global activeTask
    global nextTime
    global registeredClients
    global clients
    global aggregate

    activeTask = None
    clients = {}
    nextTime = 0
    validKeys = []
    taskId == 0
    registeredClients = 0
    aggregate  = False
    changeStep(0)


@app.route("/demo")
def demo():
    """
    Phase Status
    ---
    summary: Runs background process with 10 clients
    """
    socketio.start_background_task(target=lambda: os.system('python Demo/demo_run.py'))
    return "Demo running with 10 clients", 200




@app.route("/spec")
def spec():
    return jsonify(swagger(app))

app.register_blueprint(swaggerui_blueprint, url_prefix=SWAGGER_URL)
app.run(host='0.0.0.0', threaded=True)