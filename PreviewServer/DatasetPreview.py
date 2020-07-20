from flask import Flask, send_file
import os
import json
from io import BytesIO
from PIL import Image, ImageDraw, ImageFont
import logging

logging.basicConfig(level=logging.DEBUG)

dataset = os.path.join("..", "Dataset", "ThermalFaceDetection")

app = Flask(__name__)

@app.route('/')
def index():
    return send_file("index.html", mimetype='text/html')

@app.route('/listFiles')
def listFiles():
    files = os.listdir(os.path.join(dataset, "Annotations"))
    return json.dumps([str(f).split(".")[0] for f in files])
    

@app.route('/getImage/<id>')
def getImage(id):
    # Load Image
    img = Image.open(os.path.join(dataset, "Images", id+".jpg"))
    size = img.size
    with open(os.path.join(dataset, "Annotations", id+".json")) as json_file:
        data = json.load(json_file)
        for obj in data["objects"]:
            bbox = [
                obj["bbox"]["xmin"] * size[1], #480
                obj["bbox"]["ymin"] * size[0], #640
                obj["bbox"]["xmax"] * size[1],
                obj["bbox"]["ymax"] * size[0]
            ]

            draw = ImageDraw.Draw(img)
            draw.rectangle(bbox, outline=(255, 0, 0), width=3)

    return serve_pil_image(img)

def serve_pil_image(pil_img):
    img_io = BytesIO()
    pil_img.save(img_io, 'JPEG', quality=70)
    img_io.seek(0)
    return send_file(img_io, mimetype='image/jpeg')

@app.route('/getThermalImage/<id>')
def getThermalImage(id):
    # Load Image
    img = Image.open(os.path.join(dataset, "ThermalImages", id+".jpg"))
    size = img.size
    with open(os.path.join(dataset, "Annotations", id+".json")) as json_file:
        data = json.load(json_file)
        for obj in data["objects"]:
            bbox = [
                obj["bbox"]["xmin"] * size[1], 
                obj["bbox"]["ymin"] * size[0],
                obj["bbox"]["xmax"] * size[1],
                obj["bbox"]["ymax"] * size[0]
            ]

            dot = (obj["meta"]["maxspotABS"][1]*8 , obj["meta"]["maxspotABS"][0]*8)
            dotcrd = (dot[0]-5, dot[1]-5, dot[0]+5, dot[1]+5)
            print(dotcrd)
            draw = ImageDraw.Draw(img)
            draw.rectangle(bbox, outline=(0, 255, 0), width=3)
            draw.ellipse(dotcrd, fill=(0,0,255))
            correction = 0.0
            font = ImageFont.truetype("C:/Windows/Fonts/arial.ttf", 16)
            draw.text(dot, "{:.2f}".format(obj["meta"]["max"]/100 - 273.15 + correction), fill=(0, 0, 255), font= font)

    return serve_pil_image(img)

@app.route('/deleteAnnotation/<id>')
def deleeteAnnoation(id):
    os.remove(os.path.join(dataset, "Images", id+".jpg"))
    os.remove(os.path.join(dataset, "ThermalImages", id+".jpg"))
    os.remove(os.path.join(dataset, "Annotations", id+".json"))
    return '', 200

if __name__== '__main__':
    app.run()
    #app.run(host='0.0.0.0')