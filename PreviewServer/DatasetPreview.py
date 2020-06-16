from flask import Flask, send_file
import os
import json
from io import BytesIO
from PIL import Image, ImageDraw
dataset = os.path.join("..", "Dataset")

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
                obj["bbox"]["xmin"] * size[1], 
                obj["bbox"]["ymin"] * size[0],
                obj["bbox"]["xmax"] * size[1],
                obj["bbox"]["ymax"] * size[0]
            ]

            draw = ImageDraw.Draw(img)
            draw.rectangle(bbox, outline=(0, 255, 0), width=3)

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

            dot = (obj["meta"]["maxspotABSNorm"][0] * size[1], obj["meta"]["maxspotABSNorm"][1] * size[0])
            dotcrd = [dot, dot]
            draw = ImageDraw.Draw(img)
            draw.rectangle(bbox, outline=(0, 255, 0), width=3)
            draw.point(dotcrd, fill=(255,192,203))
            draw.text(dot, str(obj["meta"]["max"]/100 - 273.15), fill=(255,192,203))

    return serve_pil_image(img)


app.run(host='0.0.0.0')