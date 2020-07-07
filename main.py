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
from concurrent.futures import ThreadPoolExecutor
from utils.ThermalImage.camera import Camera as ThermalCamera
from utils.Tensorflow.detection import FaceDetection
from utils.OpenCV.postprocess import drawBoxes
from utils.OpenCV.camera import Camera
from utils.ThermalImage.postprocess import dataToImage
from utils.Dataset.annotation import jsonAnnotaions
from utils.fps import FPS
from io import BytesIO
#Log
logging.basicConfig(level=logging.INFO)

#Dataset config
dataset = os.path.join(os.getcwd(), "Dataset")
defaultGap = 20
gap = 2 # every x image will be added to the Dataset
resetGap = 50 # Resets the Gap when nothing happens for x frames

outputSize = (80*8, 60*8)

delay_count = 0

# Init FPS
fps = FPS()


# Init Thermalcamera
tc = ThermalCamera(uid="LcN")
tc.isTempImage()
thermalImg = None
termalfaces = []

# Init OpenCV
c = Camera()
faceDetection = FaceDetection()

# Image queue to delay thermal Image
data_queue = []
delay = 0
tick = 0

c1, c2 = 0, 0

def thermalImageThread():
    #get Data
    image_data = tc.getTemperatureImage()
    logging.debug(type(image_data))

    #get Image
    thermalImg = dataToImage(image_data, outputSize)

    #predict faces
    faceDetection.prepareImage(thermalImg, 1)
    faces = []#faceDetection.detectFace()
    # Add data to Q
    data_queue.append((thermalImg, faces, image_data))

def imageThread():
    # Capture frame-by-frame
    frame = c.takeImage()
    # Crop image from PiCam2
    frame, size = c.cropFrame(frame)
     #Detect Faces
    m = 1  #Multiplicator to reduce model imput size
    faceDetection.prepareImage(frame, m)
    faces = faceDetection.detectFace()
    return frame, size, faces

# sample image
frame = c.takeImage()
oSize = c.shape #Get the size of the original Image

thermalThread = None

with ThreadPoolExecutor(max_workers=4) as executor:
    while True:
        print(".")
        #print(f'Tasks: {executor.getActiveCount()}')
        if thermalThread is None:
            print("start")
            thermalThread = executor.submit(thermalImageThread)
        if thermalThread.done():
            print("done")
            thermalThread = None
            #thermalThread = executor.submit(getThermalImage) #Take Thermal Image and add it to the Q
        print("start image")
        ImageThread = executor.submit(imageThread)
        frame, size, faces = ImageThread.result()
    
        #thermalFaces = faceDetection.detectFace(thermal=True)
        data = None

        # Transform and Delay Thermal Image to synchronice with OpenCV Image (Delay ~1s)
        if len(data_queue) > delay:
            c2 += 1
            tick = 0
            x = data_queue.pop(0)
            thermalImg, thermalFaces, data = x
            #cv2.imshow('ThermalVideo', thermalImg)

        c1 += 1

        # Prepare OpenCV Image
        frame = cv2.resize(frame, outputSize , interpolation = cv2.INTER_AREA)

        prevFrame = frame.copy()
        # Draw a rectangle around the faces
        prevFrame = drawBoxes(prevFrame, faces)

        if thermalImg is not None:
            prevFrame = drawBoxes(prevFrame, thermalFaces, (0,0,255))
            alpha = 0.6
            prevFrame = cv2.addWeighted(prevFrame, alpha, thermalImg, 1-alpha, 0.0)

        cv2.putText(prevFrame, f'FPS: {int(fps)}', (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0))
        print(int(fps))

        # Display the resulting frame
        cv2.imshow('Video', prevFrame)

        resetGap = resetGap - 1
        if resetGap <= 0:
            gap = 2
            resetGap = 50

        fps.tick()

        print (f'{c1} / {c2}')

        #out.write(frame)
        if cv2.waitKey(10) & 0xFF == ord('q'):
            break


# When everything is done, release the capture
cv2.destroyAllWindows()
tc.close()