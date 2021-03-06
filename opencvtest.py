import cv2
import sys
import time
import numpy
import math
from PIL import Image, ImageOps
from tinkerforge.ip_connection import IPConnection
from tinkerforge.bricklet_thermal_imaging import BrickletThermalImaging

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

def cb_linear_temperature_image(image):
    # Save image to queue (for loop below)
    print("New Image")
    print(image)
    newThermal = image

# Setup Opencv
cascPath = "class.xml"
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
width, height = frame.shape[:2]
size = ( height, width)

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
    width, height = frame.shape[:2]
    size = (width, height)
    #inc Brighness for Pi Cam2
    brt = 30
    #frame[frame < 255-brt] += brt
    
    #size reduce
    m = 4
    
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray = cv2.cv2.resize(gray, (int(height / m), int(width / m)))
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

    # Draw a rectangle around the faces
    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x*m, y*m), (x*m+w*m, y*m+h*m), (0, 255, 0), 2)
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
        image = Image.new('P', (80, 60))
        image.putdata(data)
        image.putpalette(get_thermal_image_color_palette())
        #print(numpy.array(image))
        image = image.resize((80*8, 60*8), Image.ANTIALIAS)
        image = ImageOps.flip(image)
        img = numpy.array(image.convert('RGB'))
        img = img[:, :, ::-1].copy() 
        cv2.imshow('ThermalVideo', img)
    
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