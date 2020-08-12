from flask import Flask, jsonify, request, send_file
import time
import random
import json
from utils.Tensorflow import tff
from flask_swagger import swagger
from flask_swagger_ui import get_swaggerui_blueprint
from utils.Tensorflow.tff import aggregateVariables
from pymongo import MongoClient
from bson.objectid import ObjectId
import logging
import os
from uuid import uuid4
import shutil

logging.basicConfig(level=logging.INFO)

app = Flask(__name__)

#client = MongoClient('localhost', 27017)
#db = client.federated_learning


data = {}

SWAGGER_URL =  '/api/docs'
swaggerui_blueprint = get_swaggerui_blueprint(
    SWAGGER_URL,
    '/spec',
    config={
        'app_name': "Federated Server"
    },
)

timeout = 6000
taskId = 0
nextTime = 0
clients = []
registeredClients = 0
validKeys = []
completedTasks = 0

@app.route('/')
def index():
    return send_file("index.html", mimetype='text/html')


@app.route('/data')
def dataset():
    return jsonify({ "filename": "Subset3.zip" }), 202

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
    if random.random() > 0.8:
        return jsonify({ "time": 0, "id": 0 }), 205
    id = random.randint(1000, 9999)
    clients.append(id)
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

    # check if request is in Timeslot (4Min Timeframe)
    if nextTime-120 <= time.time() <= nextTime+120:
        if client_id in clients:
            clients.remove(client_id)
            validKeys.append(client_id)
            #plan = db.plans.find_one({"_id": ObjectId(str(taskId))})
            plan = json.load(open(os.path.join("tasks", f"{taskId}.json")))
            if plan is None:
                logging.info(f"cant find Task {taskId}")
                return "cant find Task", 404
            #Create folder for client results
            os.makedirs(os.path.join("checkpoints", plan['Task'], str(client_id)))
            return jsonify({ "Accepted": True, "Key": client_id, "Task": plan['Task'], "Data": plan['Data'], "Case": plan['Case'], "ModelVersion": plan['ModelVersion'] }), 200

    return jsonify({ "Accepted": False }), 410


@app.route('/model/<id>')
def updateModel(id):
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

    modelDir = os.path.join("TrainData", "model", "FederatedModel")
    checkpoints = sorted(os.listdir(modelDir))[-3:] # Should be the requested checkpoints
    zip_file = os.path.join(tempDir, 'ckpt.zip')
    with ZipFile(zip_file, 'w') as zip:
        for c in checkpoints:
            zip.write(os.path.join(modelDir, c))
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

    if nextTime <= time.time() <= nextTime+timeout:
        if id in validKeys :
            validKeys.remove(id)
            # check if the post request has the file part
            if not 'file' in request.files:
                return "", 404
            file = request.files['file']
            # if user does not select file, browser also
            # submit an empty part without filename
            if file.filename == '':
                flash('No selected file')
                return "", 404
            plan = json.load(open(os.path.join("tasks", f"{taskId}.json")))
            if plan is None:
                logging.info(f"cant find Task {taskId}")
                return "cant find Task", 404
            if file:
                try:
                    zipFile = os.path.join("checkpoints", plan['Task'], str(id), 'results.zip')
                    file.save(zipFile)
                    with zipfile.ZipFile(zipFile, 'r') as zip_ref:
                        zip_ref.extractall()
                    os.remove(zipFile)
                    return "", 200
                except:
                    return "", 500

        return "", 400
    else:
        return "", 410

@app.route('/meta/<int:id>', methods=['POST'])
def meta(id):
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
    trainresult = request.json
    print(f'JSON: {trainresult}')
    with open(os.path.join("checkpoints", plan['Task'], str(id), f"meta.json"), 'w') as json_file:
            json.dump(trainresult, json_file)
    return "", 200


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


        #dbId = db.plans.insert_one(plan).inserted_id
        taskId = uuid4()
        output = os.path.join("checkpoints", plan['Task'])
        if os.path.exists(output):
            shutil.rmtree(output)
        os.makedirs(output)

        with open(os.path.join(output, f"definition.json"), 'w') as json_file:
            json.dump(plan, json_file)

        with open(os.path.join("tasks", f"{taskId}.json"), 'w') as json_file:
            json.dump(plan, json_file)
        nextTime = utime
        return f"Plan saved with the id {taskId}", 200
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

        oldPlan = json.load(open(os.path.join("tasks", f"{taskId}.json")))
        oldPlan.pop("_id", None)

        for k in plan.keys():
            oldPlan[k] = plan[k]
        taskId = uuid4()
        with open(os.path.join("tasks", f"{taskId}.json"), 'w') as json_file:
            json.dump(plan, json_file)
        nextTime = plan['Time']
        return f"Plan saved with the id {taskId}", 200



@app.route('/aggregate')
def aggregateResults():
    """
    Endpoint that aggregates all Checkpoints from this TrainingsPhase
    ---
    tags:
        - internal
        - reporting
        - FederatedProcess
    summary: Endpoint to aggregate Trainingresults
    description: Internal Endpoint that gets triggered after the timeout to aggregate all incomming Training result to a new Checkpointfile
    produces:
    - CheckpointFile
    responses:
        200:
            description: Input Accepted
    """
    plan = json.load(open(os.path.join("tasks", f"{taskId}.json")))
    taskDir = os.path.join("checkpoints", plan['Task'])
    results = os.listdir(taskDir)
    chkps = []
    for i, result in enumerate(results):
        c = tf.train.latest_checkpoint(result)
        pipeline = os.path.join(result, 'pipeline.config')
        chkps.append(readCheckpointValues((pipeline, c , i)))

    t3 = time.time()
    logging.info(f'Load Ref: {t2-t1}, Load Checkpoints: {t3-t2}')
    aggregatedData = aggregateVariables(chkps)
    t4 = time.time()
    logging.info(f'Load Ref: {t2-t1}, Load Checkpoints: {t3-t2}, Modify Key: {t4-t3}')

    output = os.path.join( "models", f"{plan['ModelVersion']}_{time.time()}" )
    plan['newModel'] = output
    # update Plan
    with open(os.path.join("tasks", f"{taskId}.json", "w")) as jsonFile:
        json.dump(plan, jsonFile)

    writeCheckpointValues(aggregatedData, pipeline, output, checkpoint)
    t5 = time.time()
    logging.info(f'Load Ref: {t2-t1}, Load Checkpoints: {t3-t2}, Modify Key: {t4-t3}, Save Checkpoint: {t5-t4}')
    return f"Checkpoint saved {checkpointId}", 200


@app.route('/eval')
def evalResults():
    plan = json.load(open(os.path.join("tasks", f"{taskId}.json")))
    #Find Labelmap
    data = plan['Data']
    case = plan['Case']
    model = plan['newModel']
    testData = os.path.join('Testdata', case)
    labelmap = None
    files = os.listdir(testData)
    for f in files:
        if f.startswith("label_map"):
            labelmap = os.path.join(testData f)
            break
    imgDir = os.path.join(testData, data)
    annoDir = os.path.join(testData, 'annotations')
    dataDir = os.path.join(testData, 'output')
    augImages, augAnnotations = augmentData(imgDir, annoDir, dataDir, split)
    tfrecordConfig = prepareTFrecord(augImages[0], augAnnotations[0], dataDir, labelmap=labelmap, annoFormat=annoformat, split=0.1)
    eval(outDir, dataDir, tfRecordsConfig=tfrecordConfig, model= model, steps=1)


## Info Endpoints
@app.route('/status')
def getStatus():
    """
    Phase Status
    ---
    summary: Endpoint that returns the Status of the current Phase
    description: Returns Status informations for a optinal UI
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
    global taskId

    if time.time() > nextTime+timeout:
        taskId == 0
    return jsonify({"ActiveTask": False if taskId==0 else True ,"RegisteredClients": registeredClients, "AcceptedClients": len(clients)+len(validKeys)+completedTasks, "ActiveTasks": len(validKeys), "CompletedTasks": completedTasks })

@app.route("/spec")
def spec():
    return jsonify(swagger(app))

app.register_blueprint(swaggerui_blueprint, url_prefix=SWAGGER_URL)
app.run(host='0.0.0.0')