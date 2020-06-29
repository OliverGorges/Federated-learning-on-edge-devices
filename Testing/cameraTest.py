import unittest
import utils.OpenCV
import time

from utils.ThermalImage.camera import Camera as ThermalCamera
from utils.ThermalImage.postprocess import dataToImage
from utils.OpenCV.detection import FaceDetection
from utils.OpenCV.camera import Camera
from utils.fps import FPS

class TestCameras(unittest.TestCase):

    outputSize = (80*8, 60*8)

    def testConnectionCamera(self):
        self.c = Camera()
        self.assertTrue(c.getConnectionState())

    def testConnectionThermalCamera(self):
        self.tc = ThermalCamera()
        self.assertTrue(tc.getConnectionState())

    def testImage(self):
        image = c.takeImage()
        self.assertTrue(image is not None)
    
    def testImageThermalHC(self):
        image = tc.getHighContrastImage()
        self.assertTrue(image is not None)

    def testImageTermalTemp(self):
        data = tc.getTemperatureImage()
        image = dataToImage(data, outputSize)
        self.assertTrue(image is not None)

    def testFPS(self):
        fps = FPS()
        testTime = time.time() + 10 #run for 10s
        while testTime > time.time():
            image = c.takeImage()
            fps.tick()
        fps.print()
        self.assertGreater(float(fps), 10.0) #Normal Cam should have min 10 fps

    def testFPSThermalHC(self):
        fps = FPS()
        testTime = time.time() + 10 #run for 10s
        while testTime > time.time():
            image = tc.getHighContrastImage()
            fps.tick()
        fps.print()
        self.assertGreater(float(fps), 5.0) #High Contrast images can have up to 9fps, 5 should be min

    def testFPSThermalTemp(self):
        fps = FPS()
        testTime = time.time() + 10 #run for 10s
        while testTime > time.time():
            data = tc.getTemperatureImage()
            image = dataToImage(data, outputSize)
            fps.tick()
        fps.print()
        self.assertGreater(float(fps), 3.0) #Temp image has mac 4.5fps, 3fps should be min

    def testFPSwithFaceDetect(self):
        fps = FPS()
        faceDetection = FaceDetection()
        testTime = time.time() + 10 #run for 10s
        while testTime > time.time():
            image = c.takeImage()
            faceDetection.prepareImage(image, 1)
            faces = faceDetection.detectFace()
            fps.tick()
        fps.print()
        self.assertGreater(float(fps), 5.0) #Facedetection should has min 5fps

    def testFPSwithThermalFaceDetectHC(self):
        fps = FPS()
        testTime = time.time() + 10 #run for 10s
        while testTime > time.time():
            image = tc.getHighContrastImage()
            faceDetection.prepareImage(image, 1)
            faces = faceDetection.detectFace()
            fps.tick()
        fps.print()
        self.assertGreater(float(fps), 5.0) #Facedetection should has min 5fps

    def testFPSwithThermalFaceDetectTemp(self):
        fps = FPS()
        testTime = time.time() + 10 #run for 10s
        while testTime > time.time():
            data = tc.getTemperatureImage()
            image = dataToImage(data, outputSize)
            faceDetection.prepareImage(image, 1)
            faces = faceDetection.detectFace()
            fps.tick()
        fps.print()
        self.assertGreater(float(fps), 3.0) #fps reduced to 3fps due the max fps of 4.5

    def testCombinedFPS(self):
        fps = FPS()
        testTime = time.time() + 10 #run for 10s
        while testTime > time.time():
            c.takeImage()
            data = tc.getTemperatureImage()
            image = dataToImage(data, outputSize)
            fps.tick()
        fps.print()
        self.assertGreater(float(fps), 3.0) 

    def testCombinedFPSwithFaceDetect(self):
        fps = FPS()
        testTime = time.time() + 10 #run for 10s
        while testTime > time.time():
            c.takeImage()
            data = tc.getTemperatureImage()
            image = dataToImage(data, outputSize)
            fps.tick()
        fps.print()
        self.assertGreater(float(fps), 1.0) #Normal Cam Should have min 10 fps

if __name__ == '__main__':
    unittest.main()