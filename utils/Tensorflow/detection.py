import os
import six.moves.urllib as urllib
import sys
import numpy as np
import tensorflow as tf
import logging

from collections import defaultdict
from io import StringIO
from matplotlib import pyplot as plt
from PIL import Image, ImageDraw
import cv2

from google.protobuf import text_format
from object_detection.utils import ops as utils_ops
from object_detection.utils import label_map_util
from object_detection.utils import visualization_utils as vis_util
from object_detection.utils import config_util
from object_detection.builders import model_builder
from utils.Tensorflow.trainer import exportFrozenGraph
from utils.Tensorflow.kerasdummy import Model

class FaceDetection():

    
    def __init__(self, model="ssd_mobilenet_v2_coco_2018_03_29", modelType="keras"):
        """
        model: model that should be Preloaded
        modelType: supported types: tflite, savedmodel, graph
        """
        modelDir = model #os.path.join('Traindata', 'model', model )
        logging.info(f'Load Model from: {modelDir}')
        self.modelType = modelType
        tf.compat.v1.enable_eager_execution

        if self.modelType == "tflite":
            model = ''
            # Checks folder for TFlite file
            for f in os.listdir(modelDir):
                if f.endswith('.tflite'):
                    model = os.path.join(modelDir, f)
                    break
            if model == '': raise Exception(f'No TFLite model found in {modelDir}')
            
            # load model
            self.interpreter = tf.lite.Interpreter(model_path=model)
            self.interpreter.allocate_tensors()
            self.input_details = self.interpreter.get_input_details()
            self.output_details = self.interpreter.get_output_details()

        elif self.modelType == "savedmodel":
            self.model = tf.saved_model.load(modelDir)
        elif self.modelType == "keras":
            tf.keras.backend.clear_session()
            checkpointFile = tf.train.latest_checkpoint(os.path.join(modelDir, "checkpoint"))

            for f in os.listdir(modelDir):
                if f.endswith(".config"):
                    pipeline = os.path.join(modelDir, f)

            configs = config_util.get_configs_from_pipeline_file(pipeline)
            detectionModel = model_builder.build(configs['model'], is_training=False)

            model = detectionModel # Add Pre and Post Processing to model
            checkpoint = tf.train.Checkpoint(model=model)
            checkpoint.restore( checkpointFile).expect_partial()
            logging.info(f'Checkpoint restored: {checkpointFile}')
            self.model = model

            
    # Convert OpenCV image for the use with Tensorflow
    def prepareImage(self, image, scale):
        logging.debug("Prepare Image")
        width, height = image.shape[:2]
        self.image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        self.image = cv2.cv2.resize(image, (int(height / scale), int(width / scale)))

        return self.image
    
    def detectFace(self, image=None, normalized=True):
        """
        Entry Class for Detection
        uses different methods based on the selected modelType
        image: imagedata
        normalized: returns the box in a normalized format, Default: True
        """
        if image is None:
            image = self.image
        #logging.debug(image)
        if self.modelType == "tflite":
            self.detections = self.run_inference_on_tflite(image)
            pass
        elif self.modelType == "savedmodel":
            self.detections = self.run_inference_on_saved_model(self.model, image)
        elif self.modelType == "keras":
            self.detections = self.run_inference_on_keras(self.model, image)
        return self.detections

    # Normale Boxes for annoationfiles
    def normalizeBoxes(self, image=None, faces=None):
        if not image:
            width, height = self.image.shape[:2]
        else:
            width, height = image.shape[:2]
        if not faces:
            faces = self.detections

        normFaces = []
        for (x, y, w, h) in faces:
            normFaces.append(
                (
                    x / width,
                    y / height,
                    (x+w) / width,
                    (y+h) / height
                )
            )
        return normFaces

    def run_inference_on_keras(self, graph, image):
        image = np.expand_dims(cv2.resize(image, (320,320)), axis=0)
        logging.info("Run inference")

        image, shapes = self.model.preprocess(tf.convert_to_tensor(image, dtype=tf.float32))
        prediction_dict = self.model.predict(image, shapes)
        results = self.model.postprocess(prediction_dict, shapes)
        #output_dict = self.model.call(image)#self.model.predict(tf.convert_to_tensor(image, dtype=tf.float32), np.array([1, 320, 320, 3], ndmin=2))

        detection_boxes =  np.squeeze(results['detection_boxes'], axis=0) 
        multiclass_scores = np.squeeze(results['detection_multiclass_scores'], axis=0) 
        classes = []
        scores =  []
        num_detection = 0
        output_dict = {}


        for i in range(len(multiclass_scores)):
            score = multiclass_scores[i]
            scores.append(np.amax(score))
            classes.append(np.where(score == np.amax(score))[0][0])
            if np.amax(score) > 0.5:
                num_detection += 1
        # get output tensor
        output_dict['detection_boxes'] = detection_boxes
        output_dict['detection_classes'] = classes
        output_dict['detection_scores'] = scores
        output_dict['num_detections'] =  num_detection
               
        return output_dict


    def run_inference_on_saved_model(self, model, image):
        image = np.asarray(image)
        # The input needs to be a tensor, convert it using `tf.convert_to_tensor`.
        
        input_tensor = np.expand_dims(image, 0)
        output_dict = self.model(input_tensor)


        num_detections = int(output_dict.pop('num_detections').numpy()[0])
        logging.debug(num_detections)
        output_dict = {key:value[0, :num_detections].numpy() 
                        for key,value in output_dict.items()}
        output_dict['num_detections'] = num_detections

        # detection_classes should be ints.
        output_dict['detection_classes'] =  output_dict['detection_classes'].astype(np.int64)
        
        output_dict['detection_scores'] =  output_dict['detection_scores'].astype(np.float64)
       
        # Handle models with masks:
        if 'detection_masks' in output_dict:
            # Reframe the the bbox mask to the image size.
            detection_masks_reframed = utils_ops.reframe_box_masks_to_image_masks(
                    output_dict['detection_masks'], output_dict['detection_boxes'],
                    image.shape[0], image.shape[1])      
            detection_masks_reframed = tf.cast(detection_masks_reframed > 0.5,
                                            tf.uint8)
            output_dict['detection_masks_reframed'] = detection_masks_reframed.numpy()
        #else:
            #output_dict['detection_boxes'] = np.squeeze(output_dict['detection_boxes']
        return output_dict

    def run_inference_on_tflite(self, img):

        img_org = img.copy()

        #prepare Image
        shape = self.input_details[0]['shape']
        img = cv2.resize(img, (shape[1], shape[2]))
        img = img.reshape(1, img.shape[0], img.shape[1], img.shape[2]) # (1, 300, 300, 3)
        img = img.astype(np.float32)


        # set input tensor
        self.interpreter.set_tensor(self.input_details[0]['index'], img)


        # run Model
        self.interpreter.invoke()
        output_dict = {}

        # Foramts results
        # 0 = detection anchor indeces ?
        # 1 = boxes
        # 2 = broken_classes
        # 3 = multiclass detection score
        # 4 = scores
        # 5 = num of detections
        # 6 = raw detection boxes
        # 7 = raw multiclass detection score

        detection_boxes = np.squeeze(self.interpreter.get_tensor(self.output_details[1]['index']), axis=0) 
        multiclass_scores = np.squeeze(self.interpreter.get_tensor(self.output_details[3]['index']), axis=0) 
        classes = []
        scores =  []
        num_detection = 0
        output_dict = {}


        for i in range(len(multiclass_scores)):
            score = multiclass_scores[i]
            scores.append(np.amax(score))
            classes.append(np.where(score == np.amax(score))[0][0])
            if np.amax(score) > 0.5:
                num_detection += 1
        # get output tensor
        output_dict['detection_boxes'] = detection_boxes
        output_dict['detection_classes'] = classes
        output_dict['detection_scores'] = scores
        output_dict['num_detections'] =  num_detection
               
        return output_dict


