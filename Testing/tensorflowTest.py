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

    def testMaskDetectionFaceDetectionAll(self):
        
        image = cv2.imread(os.path.join("Testing", "sampleData", "ThermalFacedetect_2.jpg"), cv2.IMREAD_COLOR)
        modelDir = os.path.join('Traindata', 'model', 'ThermalModel40k')
        detector = FaceDetection(modelDir, "keras")
        detector.prepareImage(image, 1)

        t1 = time.time()
        detections = detector.detectFace()
        predTime = time.time() - t1
        logging.warning(f'### Model: {modelDir}, Predictions: {detections["num_detections"]}, Time: {predTime}')


        result = drawBoxes(image, detections)
        cv2.imwrite(os.path.join("Testing", "sampleData","output1.jpg"), result)
    
    def xtestMaskDetectionFaceDetectionAll1(self):
        
        image = cv2.imread(os.path.join("Testing", "sampleData", "faceDetection_norm_2.jpg"), cv2.IMREAD_COLOR)
        modelDir = os.path.join('graphmod', 'output', 'saved_model')
        detector = FaceDetection(modelDir, "savedmodel")
        detector.prepareImage(image, 1)
        detections = detector.detectFace()
        print(detections)
    
        result = drawBoxes(image, detections)
        cv2.imwrite(os.path.join("Testing", "sampleData","output1.jpg"), result)
        #self.assertEqual(2, 2)

    def xtestconvertSavedModel(self):
        exportFrozenGraph(os.path.join('Traindata','model','graphmod'))

    def xtestTfliteConverter(self):
        modelDir = os.path.join('Traindata','model','graphmod')
        output = os.path.join(modelDir, 'tflite')
        convertModel(modelDir, output)

    def xtestTfliteConverter(self):
        modelDir = os.path.join('Traindata','model','maskdetect')
        model = tf.saved_model.load(os.path.join(modelDir, "saved_model"))
        concFunc = model.signatures[tf.saved_model.DEFAULT_SERVING_SIGNATURE_DEF_KEY]
        concFunc.inputs[0].set_shape([1,300,300,3])
        converter = tf.lite.TFLiteConverter.from_concrete_functions([concFunc])
        
        converter.allow_custom_ops=True
        converter.optimizations = [tf.lite.Optimize.DEFAULT]
        converter.target_spec.supported_op = [tf.lite.OpsSet.TFLITE_BUILTINS, tf.lite.OpsSet.SELECT_TF_OPS]
        converter.inference_type = tf.float32
        converter.post_training_quantize = True
        #converter.inference_input_type = tf.int8
        #converter.inference_output_type = tf.float32
        converter.experimental_new_converter = False
        tflite_model = converter.convert()

        if not os.path.exists(os.path.join(modelDir, "tflite")):
            os.makedirs(os.path.join(modelDir, "tflite"))
        with tf.io.gfile.GFile(os.path.join(modelDir, "tflite", "model.tflite"), 'wb') as f:
            f.write(tflite_model)

    def xtestTfliteDection(self):
        image = cv2.imread(os.path.join("Testing", "sampleData", "faceDetection_norm_2.jpg"), cv2.IMREAD_COLOR)
        modelDir = os.path.join('Traindata', 'model', 'graphmod')
        output = os.path.join(modelDir, 'tflite')
        #convertModel(modelDir, output)

        detector = FaceDetection( output, 'tflite')
        detector.prepareImage(image, 1)
        detections = detector.detectFace()
        result = drawBoxes(image, detections)
        cv2.imwrite(os.path.join("Testing", "sampleData","outputLite.jpg"), result)

    def testTfliteDection2(self):
        image = cv2.imread(os.path.join("Testing", "sampleData", "ThermalFacedetect_2.jpg"), cv2.IMREAD_COLOR)
        modelDir = os.path.join('Traindata', 'model', 'ThermalModel40k')
        output = os.path.join(modelDir, 'tflite')
        #convertModel(modelDir, output)

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