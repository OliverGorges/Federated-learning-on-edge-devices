import cv2
import sys
import time
import numpy
import math
from uuid import uuid4
import os
import json
from copy import deepcopy
from PIL import Image, ImageOps
from tinkerforge.ip_connection import IPConnection
from tinkerforge.bricklet_thermal_imaging import BrickletThermalImaging

#Dataset config
dataset = os.path.join(os.getcwd(), "Dataset")
default_gap = 20
gap = 0 # every x image will be added to the Dataset
reset_gap = 50 # Resets the Gap when nothing happens for x frames

outputSize = (80*8, 60*8)

#Tinkerforge Config
HOST = "localhost"
PORT = 4223
UID = "LcN"
newThermal = None
thermalImg = None
#Setup Tinkerforge
print("start Connection")
ipcon = IPConnection() # Create IP connection
ti = BrickletThermalImaging(UID, ipcon) # Create device object
ipcon.connect(HOST, PORT) # Connect to brickd
while ipcon.get_connection_state() == 2 :
    print(".")
print(ipcon.get_connection_state())

def get_thermal_image_color_palette():
    palette = []

    for x in range(256):
        x /= 255.0
        palette.append(int(round(255*math.sqrt(x))))                  # RED
        palette.append(int(round(255*pow(x, 3))))                     # GREEN
        if math.sin(2 * math.pi * x) >= 0:
            palette.append(int(round(255*math.sin(2 * math.pi * x)))) # BLUE
        else:
            palette.append(0)

    return palette


# Setup Opencv
cascPath = "class.xml"
faceCascade = cv2.CascadeClassifier(cascPath)
video_capture = cv2.VideoCapture(0)

# Image queue to delay thermal Image
queue = []
delay = 5
tick = 0

fps = 0
count = 0
start = time.time()

# sample image
ret, frame = video_capture.read()
oWidth, oHeight = frame.shape[:2] #Original Size
oSize = ( oHeight, oWidth)

#out = cv2.VideoWriter('demoVideo.avi',cv2.VideoWriter_fourcc(*'DIVX'), 25, size)

ti.set_image_transfer_config(ti.IMAGE_TRANSFER_MANUAL_TEMPERATURE_IMAGE)
ti.set_resolution(ti.RESOLUTION_0_TO_655_KELVIN)
time.sleep(0.5)
#ti.register_callback(ti.CALLBACK_HIGH_CONTRAST_IMAGE, cb_linear_temperature_image)

while True:

    #request termalimage
    image_data = ti.get_temperature_image()
    queue.append(image_data)


    # Capture frame-by-frame
    ret, frame = video_capture.read()
    frame = cv2.flip(frame, 0)

    # Crop image from PiCam2
    frame = frame[int(oWidth*0.2):int(oWidth*0.95), int(oHeight*0.15):int(oHeight*0.9)] 
    width, height = frame.shape[:2]
    size = (width, height)

    #increase Brighness for Pi Cam2
    brt = 30
    #frame[frame < 255-brt] += brt
    
    #Multiplicator to reduce model imput size
    m = 1
    

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray = cv2.cv2.resize(gray, (int(height / m), int(width / m)))
    gWidth, gHeight = gray.shape[:2]
    faces = faceCascade.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(30, 30),
        flags=cv2.CASCADE_SCALE_IMAGE
    )

    # FPS Calculation
    end = time.time()
    if (end - start > 2):
        fps = int(count / 2)
        start = time.time()
        count = 0
        print(fps)
    else:
        count += 1
    

    # Normalize Boxes
    nFaces = []
    for (x, y, w, h) in faces:
        nFaces.append(
            (
                x / gWidth,
                y / gHeight,
                (x+w) / gWidth,
                (y+h) / gHeight
            )
        )
    faces = nFaces

    data = None

    # Transform and Delay Thermal Image to synchronice with OpenCV Image (Delay ~1s)
    if len(queue) > delay:
        tick = 0
        data = queue.pop(0)
        print(type(data))
        # Transform 16Bit data to 8 Bit
        img = numpy.asarray(deepcopy(data))
        img = img.reshape(60, 80)
        print(img.shape)
        thermalImg = cv2.normalize(img, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
        thermalImg = cv2.flip(thermalImg, 0)
        thermalImg =cv2.cv2.resize(thermalImg, (outputSize))
        thermalImg = cv2.applyColorMap(thermalImg, cv2.COLORMAP_AUTUMN)
        cv2.imshow('ThermalVideo', thermalImg)
    

    # Prepare OpenCV Image
    frame = cv2.resize(frame, outputSize , interpolation = cv2.INTER_AREA) 

    prevFrame = frame.copy()
    # Draw a rectangle around the faces
    for (x, y, w, h) in faces:
        x = int(x * outputSize[1])
        y = int(y * outputSize[0])
        w = int(w * outputSize[1])
        h = int(h * outputSize[0])
        cv2.rectangle(prevFrame, (x, y), (w, h), (0, 255, 0), 2)


    cv2.putText(prevFrame, f'FPS: {int(fps)}', (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0))
    
    
    # Display the resulting frame
    cv2.imshow('Video', prevFrame)
    
    #Save Data 
    if len(faces) > 0 and gap <= 0 :
        gap = default_gap
        reset_gap = 50
        uuid = uuid4()
        uuid = str(uuid)
        
        # Create Annotationset
        annotation = {}
        annotation["_id"] = uuid
        annotation["image"] = uuid + ".jpg"
        annotation["thermalImage"] = uuid + ".jpg"
        
        #Save Images
        cv2.imwrite(os.path.join(dataset, "Images", uuid + ".jpg"), frame)
        cv2.imwrite(os.path.join(dataset, "ThermalImages", uuid + ".jpg"), thermalImg)
        
        # Prepare Temperature Data
        data = numpy.asarray(data)
        data = data.reshape(60, 80)

        # Create object list
        objects = []
        for (x, y, w, h) in faces:
            face = {}
            face["type"] = "Face"

            # Map Bounding Box
            bbox = {}
            bbox["xmin"] = x
            bbox["ymin"] = y
            bbox["xmax"] = w
            bbox["ymax"] = h
            face["bbox"] = bbox
            
            # Add Metadata
            meta = {}
            rawdata = data[int(y*60):int(h*60), int(x*80):int(w*80)]
            meta["rawData"] = rawdata.tolist()
            meta["median"] = int(numpy.median(meta["rawData"]))
            meta["max"] = int(numpy.amax(meta["rawData"]))
            meta["min"] = int(numpy.amin(meta["rawData"]))
            maxspot = numpy.where(rawdata == meta["max"])
            meta["maxspot"] = [int(maxspot[0]), int(maxspot[1])]
            meta["maxspotABS"] = [int(y*60+maxspot[0]), int(x*80+maxspot[1])]
            meta["maxspotABSNorm"] = [int((y*60+maxspot[0])/60), int((x*80+maxspot[1])/80)]
            meta["maxavg5"] = int(numpy.median(data[int(max(0, meta["maxspot"][0]-2)):int(min(60, meta["maxspot"][0]+2)), int(max(0, meta["maxspot"][1]-2)):int(min(80, meta["maxspot"][1]+2))])) # Average Temperature around the maxSpot with padding 10
            face["meta"] = meta
            print(meta)
            objects.append(face)
        annotation["objects"] = objects
        
        with open(os.path.join(dataset, "Annotations", uuid +".json"), 'w') as outfile:
            json.dump(annotation, outfile)
        
    else:
        gap = gap - 1

    reset_gap = reset_gap - 1
    if reset_gap <= 0:
        gap = 0
        reset_gap = 50
    
    #out.write(frame)
    if cv2.waitKey(10) & 0xFF == ord('q'):
        break

#Save the video
#out.release()

# When everything is done, release the capture
#video_capture.release()
cv2.destroyAllWindows()
ipcon.disconnect()
