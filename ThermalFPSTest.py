## Thermal FPS Test

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


fps = 0
count = 0


ti.set_image_transfer_config(ti.IMAGE_TRANSFER_MANUAL_HIGH_CONTRAST_IMAGE)
time.sleep(0.5)


start = time.time()
while True:

    #request termalimage
    image_data = ti.get_high_contrast_image()
    
    # Display Thermal Image
    image = Image.new('P', (80, 60))
    image.putdata(image_data)
    image.putpalette(get_thermal_image_color_palette())
    #print(numpy.array(image))
    image = image.resize((80*8, 60*8), Image.ANTIALIAS)
    image = ImageOps.flip(image)
    img = numpy.array(image.convert('RGB'))
    img = img[:, :, ::-1].copy()
    
    
    ### FPS
    end = time.time()
    if (end - start > 2):
        fps = int(count / 2)
        start = time.time()
        count = 0
        print(fps)
    else:
        count += 1
    cv2.putText(img, f'FPS: {int(fps)}', (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0))
    
    #Show Image
    cv2.imshow('ThermalVideo', img)
    
    
    if cv2.waitKey(10) & 0xFF == ord('q'):
        break

cv2.destroyAllWindows()
ipcon.disconnect()
