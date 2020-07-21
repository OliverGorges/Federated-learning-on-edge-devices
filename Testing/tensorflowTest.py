import unittest
import cv2
import time
import logging
import os 
from utils.Tensorflow.detection import FaceDetection
from utils.Tensorflow.postprocess import drawBoxes

logging.basicConfig(level=logging.DEBUG)

class TestObjectDetection(unittest.TestCase):

    def testThermalFaceDetectionAll(self):
        for f in os.listdir(os.path.join("Dataset", "ThermalFaceDetection", "ThermalImages")):
            image = cv2.imread(os.path.join("Dataset", "ThermalFaceDetection", "ThermalImages", f),cv2.IMREAD_COLOR)
            cv2.waitKey(0)
            detector = FaceDetection(model="FaceDetect10k", modelType="graph")
            detector.prepareImage(image, 1)
            faces = detector.detectFace()
            logging.debug(faces)
            result = drawBoxes(image, faces) 
            cv2.imshow("Mask", result)
            cv2.waitKey(1)
        self.assertEqual(faces['num_detections'], 2)

    def testThermalFaceDetection(self):
        image = cv2.imread(os.path.join("Testing", "sampleData", "faceDetection_thermal_1_1.jpg"),cv2.IMREAD_COLOR)
        cv2.waitKey(0)
        detector = FaceDetection(model="FaceDetect10k", modelType="graph")
        detector.prepareImage(image, 1)
        faces = detector.detectFace()
        logging.debug(faces)
        result = drawBoxes(image, faces) 
        cv2.imshow("Mask", result)
        cv2.waitKey(1)
        self.assertEqual(faces['num_detections'], 2)

    def testMaskDetection(self):
        image = cv2.imread(os.path.join("Testing", "sampleData", "maskDetection_norm_2.png"),cv2.IMREAD_COLOR)
        cv2.waitKey(0)
        detector = FaceDetection(model="MaskDetect20k", modelType="graph")
        detector.prepareImage(image, 1)
        faces = detector.detectFace()
        logging.debug(faces)
        result = drawBoxes(image, faces) 
        cv2.imshow("Mask", result)
        cv2.waitKey(1)
        self.assertEqual(faces['num_detections'], 2)

    def testMaskDetectionLite(self):
        image = cv2.imread(os.path.join("Testing", "sampleData", "maskDetection_norm_2.png"),cv2.IMREAD_COLOR)
        cv2.waitKey(0)
        detector = FaceDetection(model="TFlite", modelType="tflite")
        detector.prepareImage(image, 1)
        faces = detector.detectFace()
        logging.debug(faces)
        result = drawBoxes(image, faces) 
        cv2.imshow("Mask Lite", result)
        cv2.waitKey(1)
        self.assertEqual(faces['num_detections'], 2)
    
    def testCocoData(self):
        image = cv2.imread(os.path.join("Testing", "sampleData", "coco_test.jpg"), cv2.IMREAD_COLOR)
        cv2.waitKey(0)
        detector = FaceDetection(model="ssd_mobilenet_v2_coco_2018_03_29", modelType="graph")
        detector.prepareImage(image, 1)
        faces = detector.detectFace()
        logging.debug(faces)
        result = drawBoxes(image, faces)
        cv2.imshow("Coco Lite", result)
        cv2.waitKey(1)
        self.assertEqual(faces['num_detections'], 2)

    def testCocoDataLite(self):
        image = cv2.imread(os.path.join("Testing", "sampleData", "coco_test.jpg"), cv2.IMREAD_COLOR)
        
        cv2.waitKey(0)
        detector = FaceDetection(model="ssdlite", modelType="tflite")
        detector.prepareImage(image, 1)
        faces = detector.detectFace()
        logging.debug(faces)
        result = drawBoxes(image, faces)
        cv2.imshow("Coco Lite", result)
        cv2.waitKey(1)
        self.assertEqual(faces['num_detections'], 2)

if __name__ == '__main__':
    unittest.main()