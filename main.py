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
from utils.OpenCV.detection import FaceDetection as FaceDetectionCV
from utils.Tensorflow.postprocess import drawBoxes
from utils.OpenCV.postprocess import drawBoxes as drawBoxesCV
from utils.OpenCV.camera import Camera
from utils.ThermalImage.postprocess import dataToImage
from utils.Dataset.annotation import jsonAnnotaions, previewAnnotaion
from utils.fps import FPS
from io import BytesIO
import multiprocessing
from flask import Flask
import json

#Log
logging.basicConfig(level=logging.DEBUG)

class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, numpy.ndarray):
            return obj.tolist()
        return json.JSONEncoder.default(self, obj)

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
model1 = os.path.join("Traindata", "model", "TFlite_Mask")
model2 = os.path.join("Traindata", "model", "TFlite_Thermal")
faceDetection1 = FaceDetectionCV()
faceDetection2 = FaceDetection(model=model2, modelType="tflite")
# Image queue to delay thermal Image
data_queue = []
delay = 0
tick = 0

c1, c2 = 0, 0

# Threads
def thermalImageThread():
    t1 = time.time()
    #get Data
    image_data = tc.getTemperatureImage()
    #get Image
    thermalImg = dataToImage(image_data, outputSize)
    logging.debug(f"Runtime ThermalThread: {time.time() - t1}")
    return thermalImg, image_data


def detectionThread(queue, inputDict, detector):
    t1 = time.time()
    detector.prepareImage(inputDict["Image"], inputDict["Scale"])
    
    inputDict["Faces"] = detector.detectFace()
    inputDict["Runtime"] = time.time() - t1
    logging.debug(f"Runtime DetectionThread:  {time.time() - t1} Detections: {inputDict['Faces']['num_detections']}")
    queue.put(inputDict)


# sample image
frame = c.takeImage()
oSize = c.shape #Get the size of the original Image

thermalThread = None
detectionThread1,  detectionThread2 = None, None 

frameDict = {}
q = multiprocessing.Queue()
with ThreadPoolExecutor(max_workers=8) as executor:
    
    
    while True:
        # Capture frame-by-frame
        liveFrame = c.takeImage()
        # Crop image from PiCam2
        liveFrame, size = c.cropImage(liveFrame)

        # Run ThermalImage Thread and first Detection Thread Parallel
        if detectionThread1 is None and thermalThread is None:
            thermalThread = executor.submit(thermalImageThread)
            detectionThread1 = multiprocessing.Process(target=detectionThread, args=(q, {"Image": liveFrame, "Size": size, "Scale": 1, "Thermal": False}, faceDetection1))
            detectionThread1.start()
        
        # Start Secound DatectionThread then ThermalImage thread is done
        if thermalThread is not None:
            if thermalThread.done():
                if  detectionThread2 is None:
                    c2 += 1
                    thermalImg, data = thermalThread.result()
                    print(f'{thermalImg.shape} / {liveFrame.shape}')
                    detectionThread2 = multiprocessing.Process(target=detectionThread, args=(q, {"Image": thermalImg, "Data": data, "Scale": 1, "Thermal": True}, faceDetection2))
                    detectionThread2.start()
                    thermalThread = None

        # Wait til both detection Threads have reulsts and combine the results
        if detectionThread1 is not None and detectionThread2 is not None:
            #detectionThread1.join(1)
            #detectionThread2.join(1)
            #print(".")
            #Weit til both detection Threads are done
            if q.qsize() >= 2:
                frameDict = {}
                print("Threads done")
                d1 = q.get()
                d2 = q.get()
                if d1["Thermal"]:
                    frameDict["PICam"] = d2
                    frameDict["ThermalCam"] = d1
                else:
                    frameDict["PICam"] = d1
                    frameDict["ThermalCam"] = d2
                detectionThread1, detectionThread2 = None, None
            else:
                pass
                #print("Still Alive ")
          
        c1 += 1
        # Display Images with detection results
        if "PICam" in frameDict:
            # Prepare OpenCV Image
            print(f'Norm: {frameDict["PICam"]["Image"].shape}')
            print(f'Therm: {frameDict["ThermalCam"]["Image"].shape}')

            frame = cv2.resize(frameDict["PICam"]["Image"], outputSize , interpolation = cv2.INTER_AREA)
            
            prevFrame = frame.copy()
            
            # Merge both images
            alpha = 0.8
            print(f'Norm2: {prevFrame.shape}')
            prevFrame = cv2.addWeighted(prevFrame, alpha, frameDict["ThermalCam"]["Image"], 1-alpha, 0.0)
            
            
            os.path.join("MainUI", "static", "prevFrame.png")
            cv2.imwrite(os.path.join("MainUI", "static", "prevFrame.png"),prevFrame)

            # Draw a rectangle around the faces
            prevFrame = drawBoxesCV(prevFrame, frameDict["PICam"]["Faces"])
            prevFrame = drawBoxes(prevFrame, frameDict["ThermalCam"]["Faces"], (0,0,255))
            
            cv2.putText(prevFrame, f'FPS: {int(fps)}', (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0))
            print(int(fps))
            
            previewAnnotaion(frameDict["ThermalCam"]["Data"], (frameDict["PICam"]["Faces"], frameDict["ThermalCam"]["Faces"]), os.path.join("MainUI", "static", "meta.json"))
            # Display the resulting frame
            cv2.imshow('Video', prevFrame)
            frameDict = {}

        resetGap = resetGap - 1
        if resetGap <= 0:
            gap = 2
            resetGap = 50

        fps.tick()

        #print (f'{c1} / {c2}')

        #out.write(frame)
        if cv2.waitKey(20) & 0xFF == ord('q'):
            break


# When everything is done, release the capture
cv2.destroyAllWindows()
tc.close()