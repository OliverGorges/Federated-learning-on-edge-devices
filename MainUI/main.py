from flask import Flask, send_file, jsonify
import os
import json
from io import BytesIO
from PIL import Image, ImageDraw, ImageFont
import logging

logging.basicConfig(level=logging.DEBUG)

dataset = os.path.join("..", "Dataset", "ThermalFaceDetection")

app = Flask(__name__, static_url_path='/static')

@app.route('/')
def index():
    return send_file(os.path.join("static", "index.html"), mimetype='text/html')

@app.route('/anno')
def getAnno():
    data = []
    with open(os.path.join("static", "meta.json"), "r") as read_file:
        data = json.load(read_file)
    return jsonify(data)
    
@app.route('/image')
def getImage():
    return send_file(os.path.join("static", "prevFrame.png"), mimetype='image/png')


def generate():
    # grab global references to the output frame and lock variables
    global outputFrame, lock
    # loop over frames from the output stream
    while True:
        # wait until the lock is acquired
        with lock:
            try:
                outputFrame = cv2.imread(os.path.join("static", "prevFrame.png"), flags=cv2.IMREAD_COLOR)
            except:
                continue
            # encode the frame in JPEG format
            (flag, encodedImage) = cv2.imencode(".jpg", outputFrame)
            # ensure the frame was successfully encoded
            if not flag:
                continue
        # yield the output frame in the byte format
        yield(b'--frame\r\n' b'Content-Type: image/jpeg\r\n\r\n' + 
            bytearray(encodedImage) + b'\r\n')

if __name__== '__main__':
    app.run()
    #app.run(host='0.0.0.0')