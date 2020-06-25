## FPS Check

import cv2
import sys
import time

from picamera.array import PiRGBArray
from picamera import PiCamera

from utils.ThermalImage.camera import Camera
from utils.OpenCV.detection import FaceDetection
from utils.OpenCV.postprocess import drawBoxes
from utils.ThermalImage.postprocess import dataToImage


outputSize = (80*8, 60*8)

if str(sys.argv[1]) == "Temp":
    # Init Thermalcamera
    tc = Camera(uid="LcN")
    tc.isTempImage()
elif str(sys.argv[1]) == "OpenCV":
    # Setup Opencv
    video_capture = cv2.VideoCapture(0)
elif str(sys.argv[1]) == "PiCam":
    camera = PiCamera()
    camera.framerate = 32
    camera.resolution = outputSize
    rawCapture = PiRGBArray(camera, size=outputSize)
time.sleep(0.1)
cascPath = "utils/OpenCV/class.xml"
faceCascade = cv2.CascadeClassifier(cascPath)

fps = 0
count = 0
start = time.time()


for frame in camera.capture_continuous(rawCapture, format="bgr", use_video_port=True):

    if str(sys.argv[1]) == "Temp":
        data = tc.getTemperatureImage()
        frame = dataToImage(data, outputSize)
    elif str(sys.argv[1]) == "PiCam":
    #    camera.capture(rawCapture, format="bgr")
        frame = rawCapture.array
        rawCapture.truncate()
        rawCapture.seek(0)
    elif str(sys.argv[1]) == "OpenCV":
        # Capture frame-by-frame
        ret, frame = video_capture.read()
        frame = cv2.flip(frame, 0)
    else:
        print("No Camera Selected")
        exit()

    width, height = frame.shape[:2]
    size = (width, height)
    if len(sys.argv) > 2 :
        if frame.all():
            #size reduce
            m = 2
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            gray = cv2.cv2.resize(gray, (int(height / m), int(width / m)))
            faces = faceCascade.detectMultiScale(
                gray,
                scaleFactor=1.1,
                minNeighbors=5,
                minSize=(30, 30),
                flags=cv2.CASCADE_SCALE_IMAGE
            )
        
            # Draw a rectangle around the faces
            for (x, y, w, h) in faces:
                cv2.rectangle(frame, (x*m, y*m), (x*m+w*m, y*m+h*m), (0, 255, 0), 2)

    end = time.time()
    if (end - start > 2):
        fps = int(count / 2)
        start = time.time()
        count = 0
        print(fps)
    else:
        count += 1

    

    cv2.putText(frame, f'FPS: {int(fps)}', (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0))
    
 
    # Display the resulting frame
    cv2.imshow('Video', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
cv2.destroyAllWindows()
