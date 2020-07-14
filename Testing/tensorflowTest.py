import unittest
import cv2
import time
import logging
import os 
from utils.Tensorflow.detection import FaceDetection


logging.basicConfig(level=logging.DEBUG)

class TestObjectDetection(unittest.TestCase):

    def testThermalFaceDetection(self):
        image = cv2.imread(os.path.join("Testing", "sampleData", "faceDetection_thermal_2.jpg"))
        detector = FaceDetection(model="TFlite", modelType="tflite")
        detector.prepareImage(image, 1)
        faces = detector.detectFace()
        logging.debug(faces)
        self.assertEqual(faces['num_detections'], 2)

    def testFaceDetection(self):
        image = cv2.imread(os.path.join("Testing", "sampleData", "faceDetection_norm_2.jpg"))
        detector = FaceDetection(model="TFlite", modelType="tflite")
        detector.prepareImage(image, 1)
        faces = detector.detectFace()
        logging.debug(faces)
        self.assertEqual(faces['num_detections'], 2)

if __name__ == '__main__':
    unittest.main()