from flask import Flask, jsonify
import time
import random
from utils.Tensorflow import tff
from flask_swagger import swagger
from flask_swagger_ui import get_swaggerui_blueprint

app = Flask(__name__)

data = {}

SWAGGER_URL =  '/api/docs'
swaggerui_blueprint = get_swaggerui_blueprint(
    SWAGGER_URL, 
    '/spec',
    config={  
        'app_name': "Federated Server"
    },
)

nextTime = 0
clients = []
registeredClients = 0
activeTasks = 0
completedTasks = 0

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
    registeredClients += 1
    if random.random() > 0.8:
        return jsonify({ "time": 0, "id": 0 }), 205
    id = random.randint(1000, 9999)
    clients.append(id)
    return jsonify({ "time": nextTime, "id": id }), 202

@app.route('/task/<int:id>')
def getTask(id):
    """
    Create a new user
    ---
    tags:
        - users
    definitions:
        - schema:
            id: Group
            properties:
            name:
                type: string
                description: the group's name
    """
    # check if request is in Timeslot
    if nextTime-1000 <= time.time() <= nextTime+1000:
        if id in clients:
            clients.remove(id)
            return jsonify({ "Accepted": True, "Task": "XYZ", "Data": "Thermal" })
    return jsonify({ "Accepted": False })

@app.route('/results', methods=['POST'])
def results():
    data = request.json()
    return 200

@app.route('/status')
def getStatus():
    return jsonify({ "RegisteredClients": registeredClients, "AcceptedClients": len(clients)+activeTasks+completedTasks, "ActiveTasks": activeTasks, "CompletedTasks": completedTasks })

@app.route("/spec")
def spec():
    return jsonify(swagger(app))

app.register_blueprint(swaggerui_blueprint, url_prefix=SWAGGER_URL)
app.run()