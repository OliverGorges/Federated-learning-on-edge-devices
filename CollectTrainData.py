import cv2
import sys
import time
import numpy
import math
from uuid import uuid4
import os
import json
import logging
from copy import deepcopy
from PIL import Image, ImageOps
from utils.ThermalImage.camera import Camera
from utils.OpenCV.detection import FaceDetection
from utils.OpenCV.postprocess import drawBoxes
from utils.ThermalImage.postprocess import dataToImage
from utils.Dataset.annotation import jsonAnnotaions
from utils.fps import FPS

#Log
logging.basicConfig(level=logging.INFO)

#Dataset config
dataset = os.path.join(os.getcwd(), "Dataset")
defaultGap = 20
gap = 0 # every x image will be added to the Dataset
resetGap = 50 # Resets the Gap when nothing happens for x frames

outputSize = (80*8, 60*8)

# Init Thermalcamera
tc = Camera(uid="LcN")
tc.isTempImage()
thermalImg = None

# Init FPS
fps = FPS()

# Init OpenCV
videoCapture = cv2.VideoCapture(0)
faceDetection = FaceDetection()

# Image queue to delay thermal Image
queue = []
delay = 5
tick = 0



# sample image
ret, frame = videoCapture.read()
oWidth, oHeight = frame.shape[:2] #Original Size
oSize = ( oHeight, oWidth)



while True:
    #request termalimage and add to image queue
    image_data = tc.getTemperatureImage()
    queue.append(image_data)


    # Capture frame-by-frame
    ret, frame = videoCapture.read()
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

    #Destect Faces
    faceDetection.prepareImage(frame, m)
    faces = faceDetection.detectFace()



    data = None

    # Transform and Delay Thermal Image to synchronice with OpenCV Image (Delay ~1s)
    if len(queue) > delay:
        tick = 0
        data = queue.pop(0)
        logging.debug(type(data))
        # Transform 16Bit data to 8 Bit
        thermalImg = dataToImage(data, outputSize)
        cv2.imshow('ThermalVideo', thermalImg)


    # Prepare OpenCV Image
    frame = cv2.resize(frame, outputSize , interpolation = cv2.INTER_AREA)

    prevFrame = frame.copy()
    # Draw a rectangle around the faces
    frame = drawBoxes(prevFrame, faces)

    cv2.putText(prevFrame, f'FPS: {int(fps)}', (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0))


    # Display the resulting frame
    cv2.imshow('Video', prevFrame)

    #Save Data
    try:
        if len(faces) > 0 and gap <= 0 :
            gap = defaultGap
            resetGap = 50
            uuid = str(uuid4())
            #Save Images
            cv2.imwrite(os.path.join(dataset, "Images", uuid + ".jpg"), frame)
            cv2.imwrite(os.path.join(dataset, "ThermalImages", uuid + ".jpg"), thermalImg)
            jsonAnnotaions(uuid, data, faces, os.path.join(dataset, "Annotations"))
    except:
        logging.warning("Cant create Annotation")
    else:
        gap = gap - 1

    resetGap = resetGap - 1
    if resetGap <= 0:
        gap = 0
        resetGap = 50

    fps.tick()

    #out.write(frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break


# When everything is done, release the capture
cv2.destroyAllWindows()
tc.ipcon.disconnect()