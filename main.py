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
from utils.Tensorflow.postprocess import drawBoxes
from utils.OpenCV.camera import Camera
from utils.ThermalImage.postprocess import dataToImage
from utils.Dataset.annotation import jsonAnnotaions
from utils.fps import FPS
from io import BytesIO
from multiprocessing import Process

#Log
logging.basicConfig(level=logging.DEBUG)

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
faceDetection1 = FaceDetection(model="TFlite", modelType="tflite")
faceDetection2 = FaceDetection(model="TFlite", modelType="tflite")
# Image queue to delay thermal Image
data_queue = []
delay = 0
tick = 0

c1, c2 = 0, 0

def thermalImageThread():
    t1 = time.time()
    #get Data
    image_data = tc.getTemperatureImage()
    #logging.debug(type(image_data))
    #get Image
    thermalImg = dataToImage(image_data, outputSize)
    logging.debug(f"Runtime ThermalThread: {time.time() - t1}")
    return thermalImg, image_data

def imageThread(frame):
    t1 = time.time()
    #Detect Faces
    t2 = time.time()
    
    logging.debug(f"Runtime ImageThread:  {time.time() - t1} / Image {t2 - t1} / Prediction {time.time() - t2}")
    return frame, size, faces

def detectionThread(inputDict, detector):
    t1 = time.time()
    detector.prepareImage(inputDict["Image"], inputDict["Scale"])
    inputDict["Faces"] = detector.detectFace()
    logging.debug(f"Runtime DetectionThread:  {time.time() - t1} ")
    return inputDict

# sample image
frame = c.takeImage()
oSize = c.shape #Get the size of the original Image

thermalThread = None
detectionThread1,  detectionThread2 = None, None 

frameDict = {}

with ThreadPoolExecutor(max_workers=4) as executor:
    while True:
        # Capture frame-by-frame
        liveFrame = c.takeImage()
        # Crop image from PiCam2
        liveFrame, size = c.cropImage(liveFrame)

        #print(f'Tasks: {executor.getActiveCount()}')
        if thermalThread is None:
            thermalThread = executor.submit(thermalImageThread)
        if thermalThread.done():
            if detectionThread1 is None and detectionThread2 is None:
                c2 += 1
                thermalImg, data = thermalThread.result()
                detectionThread1 = executor.submit(detectionThread, {"Image": liveFrame, "Size": size, "Scale": 1}, faceDetection1)
                detectionThread2 = executor.submit(detectionThread, {"Image": thermalImg, "Data": data, "Scale": 1}, faceDetection2)
                thermalThread = None


        if detectionThread1 is not None and detectionThread2 is not None:
            if detectionThread1.done() and detectionThread2.done():
                frameDict = {}
                frameDict["PICam"] = detectionThread1.result()
                frameDict["ThermalCam"] = detectionThread2.result()
                detectionThread1, detectionThread2 = None, None
          
        c1 += 1

        if "PICam" in frameDict:
            # Prepare OpenCV Image
            frame = cv2.resize(frameDict["PICam"]["Image"], outputSize , interpolation = cv2.INTER_AREA)

            prevFrame = frame.copy()
            # Draw a rectangle around the faces
            prevFrame = drawBoxes(prevFrame, frameDict["PICam"]["Faces"])

            
            prevFrame = drawBoxes(prevFrame, frameDict["ThermalCam"]["Faces"], (0,0,255))
            alpha = 0.6
            prevFrame = cv2.addWeighted(prevFrame, alpha, frameDict["ThermalCam"]["Image"], 1-alpha, 0.0)

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