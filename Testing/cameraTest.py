import unittest
import cv2
import time
import logging

from utils.ThermalImage.camera import Camera as ThermalCamera
from utils.ThermalImage.postprocess import dataToImage
from utils.OpenCV.detection import FaceDetection
from utils.OpenCV.camera import Camera
from utils.fps import FPS


logging.basicConfig(level=logging.INFO)

class TestCameras(unittest.TestCase):

    def testConnectionCamera(self):
        c = Camera()
        self.assertTrue(c.getConnectionState())
        c.close()

    def testConnectionThermalCamera(self):
        tc = ThermalCamera()
        self.assertTrue(tc.getConnectionState())
        tc.close()

    def testImage(self):
        c = Camera()
        image = c.takeImage()
        self.assertTrue(image is not None)
        c.close()

    
    def testImageThermalHC(self):
        tc = ThermalCamera()
        image = tc.getHighContrastImage()
        self.assertTrue(image is not None)
        tc.close()

    def testImageTermalTemp(self):
        tc = ThermalCamera()
        data = tc.getTemperatureImage()
        outputSize = (80*8, 60*8)
        image = dataToImage(data, outputSize)
        self.assertTrue(image is not None)
        tc.close()

    def testFPS(self):
        fps = FPS()
        c = Camera()
        testTime = time.time() + 10 #run for 10s
        while testTime > time.time():
            image = c.takeImage()
            fps.tick()
        logging.info(f'FPS: {float(fps)}')
        self.assertGreater(float(fps), 10.0) #Normal Cam should have min 10 fps
        c.close()

    def testFPSThermalHC(self):
        fps = FPS()
        tc = ThermalCamera()
        testTime = time.time() + 10 #run for 10s
        while testTime > time.time():
            data = tc.getHighContrastImage()
            fps.tick()
        logging.info(f'HighContrast FPS: {float(fps)}')
        self.assertGreater(float(fps), 5.0) #High Contrast images can have up to 9fps, 5 should be min
        tc.close()

    def testFPSThermalTemp(self):
        fps = FPS()
        tc = ThermalCamera()
        outputSize = (80*8, 60*8)
        testTime = time.time() + 10 #run for 10s
        while testTime > time.time():
            data = tc.getTemperatureImage()
            image = dataToImage(data, outputSize)
            fps.tick()
        logging.info(f'Temp FPS: {float(fps)}')
        self.assertGreater(float(fps), 3.0) #Temp image has mac 4.5fps, 3fps should be min
        tc.close()

    def testFPSwithFaceDetect(self):
        fps = FPS()
        c = Camera()
        faceDetection = FaceDetection()
        testTime = time.time() + 10 #run for 10s
        while testTime > time.time():
            image = c.takeImage()
            faceDetection.prepareImage(image, 1)
            faces = faceDetection.detectFace()
            fps.tick()
        logging.info( f'FPS (FaceDetect): {float(fps)}')
        self.assertGreater(float(fps), 5.0) #Facedetection should has min 5fps
        c.close()

    def testFPSwithThermalFaceDetectHC(self):
        fps = FPS()
        tc = ThermalCamera()
        outputSize = (80*8, 60*8)
        faceDetection = FaceDetection()
        testTime = time.time() + 10 #run for 10s
        while testTime > time.time():
            data = tc.getHighContrastImage()
            image = dataToImage(data, outputSize)
            faceDetection.prepareImage(image, 1)
            faces = faceDetection.detectFace()
            fps.tick()
        logging.info(f'HighContrast FPS (FaceDetect): {float(fps)}')
        self.assertGreater(float(fps), 5.0) #Facedetection should has min 5fps
        tc.close()

    def testFPSwithThermalFaceDetectTemp(self):
        fps = FPS()
        tc = ThermalCamera()
        faceDetection = FaceDetection()
        outputSize = (80*8, 60*8)
        testTime = time.time() + 10 #run for 10s
        while testTime > time.time():
            data = tc.getTemperatureImage()
            image = dataToImage(data, outputSize)
            faceDetection.prepareImage(image, 1)
            faces = faceDetection.detectFace()
            fps.tick()
        logging.info( f'Temp FPS (FaceDetect): {float(fps)}')
        self.assertGreater(float(fps), 3.0) #fps reduced to 3fps due the max fps of 4.5
        tc.close()

    def testCombinedFPS(self):
        fps = FPS()
        tc = ThermalCamera()
        c = Camera()
        outputSize = (80*8, 60*8)
        testTime = time.time() + 10 #run for 10s
        while testTime > time.time():
            c.takeImage()
            data = tc.getTemperatureImage()
            image = dataToImage(data, outputSize)
            fps.tick()
        logging.info( f'CombindedFPD: {float(fps)}')
        self.assertGreater(float(fps), 3.0) 
        tc.close()
        c.close()

    def testCombinedFPSwithFaceDetect(self):
        fps = FPS()
        tc = ThermalCamera()
        c = Camera()
        outputSize = (80*8, 60*8)
        testTime = time.time() + 10 #run for 10s
        while testTime > time.time():
            c.takeImage()
            data = tc.getTemperatureImage()
            image = dataToImage(data, outputSize)
            fps.tick()
        logging.info( f'CombindedFPS (FaceDetect): {float(fps)}')
        self.assertGreater(float(fps), 1.0) #Normal Cam Should have min 10 fps
        tc.close()
        c.close()

if __name__ == '__main__':
    unittest.main()