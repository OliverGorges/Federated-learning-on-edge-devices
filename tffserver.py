from flask import Flask, jsonify, request, send_file
import time
import random
from utils.Tensorflow import tff
from flask_swagger import swagger
from flask_swagger_ui import get_swaggerui_blueprint
from utils.Tensorflow.tff import aggregateVariables
from pymongo import MongoClient
from bson.objectid import ObjectId
import logging
from uuid import uuid4

logging.basicConfig(level=logging.INFO)

app = Flask(__name__)

client = MongoClient('localhost', 27017)
db = client.federated_learning

data = {}

SWAGGER_URL =  '/api/docs'
swaggerui_blueprint = get_swaggerui_blueprint(
    SWAGGER_URL,
    '/spec',
    config={
        'app_name': "Federated Server"
    },
)

timeout = 600
taskId = 0
nextTime = 0
clients = []
registeredClients = 0
validKeys = []
completedTasks = 0

@app.route('/')
def index():
    return send_file("index.html", mimetype='text/html')

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
            plan = db.plans.find_one({"_id": ObjectId(str(taskId))})
            if plan is None:
                logging.info(f"cant find Task {taskId}")
                return "cant find Task", 404
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
        if id in validKeys:
            validKeys.remove(id)
            trainresult1 = request.json
            trainresult2 = request.data
            print(f'Data: {trainresult2}, JSON: {trainresult1}')
            data[id] = trainresult1
            completedTasks += 1
            return 200
    else:
        return 410

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
        utime = int(plan['Time'])
        if utime  < time.time():
            logging.info("Timeerror")
            #return "Time not valid", 410
            utime = time.time()+20

        dbId = db.plans.insert_one(plan).inserted_id
        taskId = dbId
        nextTime = utime
        return f"Plan saved with the id {dbId}", 200
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

        oldPlan = db.plans.find_one({"_id": ObjectId(str(plan['ID']))})
        oldPlan.pop("_id", None)
        for k in plan.keys():
            oldPlan[k] = plan[k]
        dbId = db.plans.insert_one(plan).inserted_id
        taskId = dbId
        nextTime = plan['Time']
        return f"Plan saved with the id {dbId}", 200



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
    aggregatedData = aggregateVariables(data)
    dbId = db.checkpoints.insert_one(aggregatedData).inserted_id
    writeCheckpointValues(aggregatedData, model_dir, os.path.join("Traindata", "model", "federatedModel", f"modi_model{time.time()}.ckpt" ))
    return f"Checkpoint saved {dbId}"


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