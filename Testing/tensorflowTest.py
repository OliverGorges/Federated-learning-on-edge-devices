import unittest
import cv2
import time
import logging
import os 
import numpy as np
import tensorflow.compat.v2 as tf
from utils.Tensorflow.detection import FaceDetection
from utils.Tensorflow.postprocess import drawBoxes
from utils.Tensorflow.trainer import exportFrozenGraph
from utils.Tensorflow.tfliteConverter import convertModel

logging.basicConfig(level=logging.INFO)

"""
TF 2.2.0 / Objectdtection 2.0 Tests
"""

class TestObjectDetection(unittest.TestCase):

    #General
    def xtestconvertSavedModel(self):
        exportFrozenGraph(os.path.join('Traindata','model','ThermalModel50K'))

    def xtestTfliteConverter(self):
        modelDir = os.path.join('Traindata','model','graphmod')
        output = os.path.join(modelDir, 'tflite')
        convertModel(modelDir, output)

    #MaskDetect
    def xtestMaskDetection(self):
        
        image = cv2.imread(os.path.join("Testing", "sampleData", "Maskdetect_2.jpg"), cv2.IMREAD_COLOR)
        modelDir = os.path.join('Traindata', 'model', 'MaskModel30k')
        detector = FaceDetection(modelDir, "keras")
        detector.prepareImage(image, 1)

        t1 = time.time()
        detections = detector.detectFace()
        predTime = time.time() - t1
        logging.warning(f'### Model: {modelDir}, Predictions: {detections["num_detections"]}, Time: {predTime}')


        result = drawBoxes(image, detections, color=(255, 255, 0))
        cv2.imwrite(os.path.join("Testing", "sampleData","output1.jpg"), result)
    
    def xtestTfliteMaskDection(self):
        image = cv2.imread(os.path.join("Testing", "sampleData", "Maskdetect_2.jpg"), cv2.IMREAD_COLOR)
        modelDir = os.path.join('Traindata', 'model', 'MaskModel30k')
        output = os.path.join(modelDir, 'tflite')
        convertModel(modelDir, output)

        detector = FaceDetection( output, 'tflite')
        detector.prepareImage(image, 1)
        t1 = time.time()
        detections = detector.detectFace()
        predTime = time.time() - t1
        logging.warning(f'### Model: {output}, Predictions: {detections["num_detections"]}, Time: {predTime}')

        result = drawBoxes(image, detections, color=(255, 255, 0))
        cv2.imwrite(os.path.join("Testing", "sampleData","outputLite1.jpg"), result)

    #ThermalDetect
    def testThermalDetection(self):
        
        image = cv2.imread(os.path.join("Testing", "sampleData", "ThermalFacedetect_2.jpg"), cv2.IMREAD_COLOR)
        modelDir = os.path.join('Traindata', 'model', 'ThermalModel70map')
        detector = FaceDetection(modelDir, "keras")
        detector.prepareImage(image, 1)

        t1 = time.time()
        detections = detector.detectFace()
        predTime = time.time() - t1
        logging.warning(f'### Model: {modelDir}, Predictions: {detections["num_detections"]}, Time: {predTime}')


        result = drawBoxes(image, detections)
        cv2.imwrite(os.path.join("Testing", "sampleData","output2.jpg"), result)
    
    def testTfliteThermalDection(self):
        image = cv2.imread(os.path.join("Testing", "sampleData", "ThermalFacedetect_2.jpg"), cv2.IMREAD_COLOR)
        modelDir = os.path.join('Traindata', 'model', 'ThermalModel70map')
        output = os.path.join(modelDir, 'tflite')
        convertModel(modelDir, output)

        detector = FaceDetection( output, 'tflite')
        detector.prepareImage(image, 1)
        t1 = time.time()
        detections = detector.detectFace()
        predTime = time.time() - t1
        logging.warning(f'### Model: {output}, Predictions: {detections["num_detections"]}, Time: {predTime}')

        
        result = drawBoxes(image, detections)
        cv2.imwrite(os.path.join("Testing", "sampleData","outputLite2.jpg"), result)


if __name__ == '__main__':
    unittest.main()