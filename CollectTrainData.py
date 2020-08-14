import cv2
import sys
import time
import numpy
import math
from uuid import uuid4
import os
import json
import logging
import threading
import concurrent.futures
from copy import deepcopy
from PIL import Image, ImageOps
from io import BytesIO

from utils.ThermalImage.camera import Camera as ThermalCamera
from utils.OpenCV.detection import FaceDetection
from utils.OpenCV.postprocess import drawBoxes
from utils.OpenCV.camera import Camera
from utils.ThermalImage.postprocess import dataToImage
from utils.Dataset.annotation import jsonAnnotaions
from utils.fps import FPS
#Log
logging.basicConfig(level=logging.INFO)

#Dataset config
dataset = os.path.join(os.getcwd(), "Dataset", "ThermalDetection2")
defaultGap = 20
gap = 2 # every x image will be added to the Dataset
resetGap = 50 # Resets the Gap when nothing happens for x frames

outputSize = (80*8, 60*8)


# Init FPS
fps = FPS()


# Init Thermalcamera
tc = ThermalCamera(uid="LcN")
tc.isTempImage()
thermalImg = None

# Init OpenCV
c = Camera()
faceDetection = FaceDetection()

# Image queue to delay thermal Image
queue = []
delay = 5
tick = 0



# sample image
frame = c.takeImage()
oSize = c.shape #Get the size of the original Image

#Threading
thermalThread = None
detectionThread = None
newData = False
def thermalWorker():
    return tc.getTemperatureImage()

def detectionWorker():
    return faceDetection.detectFace()

time.sleep(1)

while True:
    start = time.time()

    frame = c.takeImage()
    
    frame, size = c.cropImage(frame)
    
    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as e:
        # Check if the Thermal Camera Therad is still running
        if thermalThread is None or thermalThread.done():
            # Get Results from the finished Threads
            if thermalThread and detectionThread:
                queue.append(thermalThread.result())
                faces = detectionThread.result()
                newData = True

            #request new thermalimage
            thermalThread = e.submit(thermalWorker)
            #image_data = tc.getTemperatureImage()
            
            #Detect Faces
            m = 1  #Multiplicator to reduce model imput size
            faceDetection.prepareImage(frame, m)
            detectionThread = e.submit(detectionWorker)
            #faces = faceDetection.detectFace()
            
            
            print('T1: future: {}'.format(thermalThread))
            print('T2: future: {}'.format(detectionThread))
            if newData:
                print(faces)
                # Transform and Delay Thermal Image to synchronice with OpenCV Image (Delay ~1s)
                data = None
                print(len(queue))
                if len(queue) > delay:
                    tick = 0
                    data = queue.pop(0)
                    logging.debug(type(data))
                    try:
                        # Transform 16Bit data to 8 Bit
                        thermalImg = dataToImage(data, outputSize)
                        #cv2.imshow('ThermalVideo', thermalImg)
                    except:
                        logging.debug("No Thermal Image Data")

                # Prepare OpenCV Image
                frame = cv2.resize(frame, outputSize , interpolation = cv2.INTER_AREA)

                prevFrame = frame.copy()
                # Draw a rectangle around the faces
                prevFrame = drawBoxes(prevFrame, faces)

                cv2.putText(prevFrame, f'FPS: {int(fps)}', (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0))
                
                #print(int(fps))
                try:
                    alpha = 0.4
                    prevFrame = cv2.addWeighted(prevFrame, alpha, cv2.resize(thermalImg, outputSize , interpolation = cv2.INTER_AREA), 1-alpha, 0.0)
                except:
                    print("Cant merge Image")
                # Display the resulting frame
                cv2.imshow('Video', prevFrame)

                #Save Data
                try:
                    if faces['num_detections'] > 0 and gap <= 0 :
                        
                        gap = defaultGap
                        resetGap = 50
                        uuid = str(uuid4())
                        #Save Images
                        cv2.imwrite(os.path.join(dataset, "Images", uuid + ".jpg"), frame)
                        cv2.imwrite(os.path.join(dataset, "ThermalImages", uuid + ".jpg"), thermalImg)
                        jsonAnnotaions(uuid, data, faces, os.path.join(dataset, "Annotations"))
                except:
                    logging.info('Can`t create Annotation')
                else:
                    gap = gap - 1

                resetGap = resetGap - 1
                if resetGap <= 0:
                    gap = 2
                    resetGap = 50
        else:
            print("skip image")
    fps.tick()
    #out.write(frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
    
    logging.debug(time.time()-start)


# When everything is done, release the capture
cv2.destroyAllWindows()
tc.close()