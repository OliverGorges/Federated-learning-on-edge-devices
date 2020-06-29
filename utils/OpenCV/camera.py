import time
import logging
import cv2

class Camera():

    def __init__(self, id:int=0):
        self.videoCapture = cv2.VideoCapture(id)
        time.sleep(0.5)

    def getConnectionState(self):
        return self.videoCapture.isOpened()

    def takeImage(self, flip:bool=True):
        ret, frame = self.videoCapture.read()
        width, height = frame.shape[:2]
        if flip:
            frame = cv2.flip(frame, 0)

        self.image = frame
        self.shape = (width, height)

        return self.image

    # Crop Frame to fit the Image from the PICam2 with ThermalImage
    def cropFrame(self, image=None, factor=[0.2, 0.95, 0.15, 0.9]):
        if isinstance(factor, float):
            factor = [factor, factor, factor, factor]
        if not image:
            image = self.image
        oWidth, oHeight = image.shape[:2]
        image = image[int(oWidth*factor[0]):int(oWidth*factor[1]), int(oHeight*factor[2]):int(oHeight*factor[3])]
        width, height = image.shape[:2]
        return image, (width, height) 
    
     #increase Brighness for Pi Cam2
    def fixBrightness(self, image, brt=30):
        return image#[image < 255-brt] += brt
    
    def close(self):
        self.videoCapture.release()

