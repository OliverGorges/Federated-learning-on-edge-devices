import cv2
import sys
import time
import numpy
import math
from PIL import Image, ImageOps
from tinkerforge.ip_connection import IPConnection
from tinkerforge.bricklet_thermal_imaging import BrickletThermalImaging

from utils.ThermalImage.postprocess import dataToImage

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

outputSize = (80*8, 60*8)


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

def cb_linear_temperature_image(image):
    # Save image to queue (for loop below)
    print("New Image")
    print(image)
    newThermal = image

# Setup Opencv
cascPath = "utils/OpenCV/class.xml"
faceCascade = cv2.CascadeClassifier(cascPath)

video_capture = cv2.VideoCapture(0)

# Image queue
queue = []
delay = 5
tick = 0

fps = 0
count = 0
start = time.time()

# sample image
ret, frame = video_capture.read()
owidth, oheight = frame.shape[:2]
#PiCam2 image Crop

#out = cv2.VideoWriter('demoVideo.avi',cv2.VideoWriter_fourcc(*'DIVX'), 25, size)

ti.set_image_transfer_config(ti.IMAGE_TRANSFER_MANUAL_HIGH_CONTRAST_IMAGE)
time.sleep(0.5)
#ti.register_callback(ti.CALLBACK_HIGH_CONTRAST_IMAGE, cb_linear_temperature_image)

while True:

    #request termalimage
    image_data = ti.get_high_contrast_image()
    queue.append(image_data)

    # Capture frame-by-frame
    ret, frame = video_capture.read()
    frame = cv2.flip(frame, 0)

    frame = frame[int(owidth*0.2):int(owidth*0.95), int(oheight*0.15):int(oheight*0.9)]

    width, height = frame.shape[:2]
    size = (width, height)
    print(size)
    #inc Brighness for Pi Cam2
    brt = 30
    #frame[frame < 255-brt] += brt

    #size reduce
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
        print(nFaces)
        faces = nFaces


    frame = cv2.resize(frame, (80*8, 60*8) , interpolation = cv2.INTER_AREA)
    # Draw a rectangle around the faces
    for (x, y, w, h) in faces:
        x = int(x * 60*8)
        y = int(y * 80*8)
        w = int(w * 60*8)
        h = int(h * 80*8)
        cv2.rectangle(frame, (x, y), (w, h), (0, 255, 0), 2)
        #cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)




    cv2.putText(frame, f'FPS: {int(fps)}', (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0))

    #frame = cv2.cv2.resize(frame, size)

    # Display the resulting frame
    cv2.imshow('Video', frame)

    #delay = fps
    #print( len(queue))

    if len(queue) > delay:
        tick = 0
        data = queue.pop(0)

        # Display Thermal Image
        thermalImg = dataToImage(data, outputSize)
        gray = cv2.cvtColor(thermalImg, cv2.COLOR_BGR2GRAY)
        gray = cv2.cv2.resize(gray, (int(height / m), int(width / m)))
        gWidth, gHeight = gray.shape[:2]
        boxes = faceCascade.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(30, 30),
            flags=cv2.CASCADE_SCALE_IMAGE
        )
        for (x, y, w, h) in boxes:
            x = int(x)
            y = int(y)
            w = int(w)
            h = int(h)
            cv2.rectangle(thermalImg, (x, y), (x+w, y+h), (0, 255, 0), 2)
        cv2.imshow('ThermalVideo', thermalImg)

    #x = input()
    #out.write(frame)
    if cv2.waitKey(10) & 0xFF == ord('q'):
        break

#Save the video
#out.release()

# When everything is done, release the capture
#video_capture.release()
#cv2.destroyAllWindows()
ipcon.disconnect()