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

logging.basicConfig(level=logging.DEBUG)

"""
TF 2.2.0 / Objectdtection 2.0 Tests
"""

class TestObjectDetection(unittest.TestCase):

    def testMaskDetectionFaceDetectionAll(self):
        
        image = cv2.imread(os.path.join("Testing", "sampleData", "maskDetection_norm_2.png"), cv2.IMREAD_COLOR)
        detector = FaceDetection('Traindata/output/test002/saved_model/', "savedmodel")
        detector.prepareImage(image, 1)
        detections = detector.detectFace()
        print(detections)
    
        result = drawBoxes(image, detections)
        cv2.imwrite(os.path.join("Testing", "sampleData","output.jpg"), result)
        #self.assertEqual(2, 2)
    
    def testconvertSavedModel(self):
        exportFrozenGraph(os.path.join('Traindata','model','maskdetect'))

    def testTfliteConverter(self):
        modelDir = os.path.join('Traindata','model','maskdetect')
        model = tf.saved_model.load(os.path.join(modelDir, "saved_model"))
        concFunc = model.signatures[tf.saved_model.DEFAULT_SERVING_SIGNATURE_DEF_KEY]
        concFunc.inputs[0].set_shape([1,300,300,3])
        converter = tf.lite.TFLiteConverter.from_concrete_functions([concFunc])
        converter.optimizations = [tf.lite.OpsSet.TFLITE_BUILTINS]
        converter.inference_type = tf.float32
        converter.post_training_quantize = True
        converter.allow_custom_ops=True
        #converter.inference_input_type = tf.int8
        #converter.inference_output_type = tf.float32
        converter.experimental_new_converter = False
        tflite_model = converter.convert()

        if not os.path.exists(os.path.join(modelDir, "tflite")):
            os.makedirs(os.path.join(modelDir, "tflite"))
        with tf.io.gfile.GFile(os.path.join(modelDir, "tflite", "model.tflite"), 'wb') as f:
            f.write(tflite_model)


if __name__ == '__main__':
    unittest.main()