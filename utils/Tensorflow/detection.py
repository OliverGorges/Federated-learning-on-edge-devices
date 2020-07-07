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

from object_detection.utils import ops as utils_ops
from object_detection.utils import label_map_util
from object_detection.utils import visualization_utils as vis_util


class FaceDetection():

    """
    model: model that should be Preloaded
    isTFLite: Selection if the used model in TF or TFLite
    """
    def __init__(self, model="ssd_mobilenet_v2_coco_2018_03_29", isTFLite=False):
        self.isTFLite = isTFLite
        modelDir = os.path.join('Traindata', 'model', model,"saved_model" )
        print(f'Load Model from: {modelDir}')
        if isTFLite:
            model = ''
            for f in os.listdir(modelDir):
                if f.endswith('.tflite'):
                    model = os.path.join(modelDir, f)
                    break
            if model == '': raise Exception(f'No TFLite model found in {modelDir}')
            
        else:
            self.model = tf.compat.v2.saved_model.load(modelDir, None)
            self.model = self.model.signatures['serving_default']
            

    def prepareImage(self, image, scale):
        logging.debug("Prepare Image")
        width, height = image.shape[:2]
        self.image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        self.image = cv2.cv2.resize(image, (int(height / scale), int(width / scale)))

        return self.image

    def detectFace(self, image=None, normalized=True):
        if not image:
            image = self.image
        logging.debug(image)
        if self.isTFLite:
            pass
        else:
            self.faces = self.run_inference_for_single_image(self.model, image)


    def normalizeBoxes(self, image=None, faces=None):
        if not image:
            width, height = self.image.shape[:2]
        else:
            width, height = image.shape[:2]
        if not faces:
            faces = self.faces

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

    def run_inference_for_single_image(self, model, image):
    
        image = np.asarray(image)
        # The input needs to be a tensor, convert it using `tf.convert_to_tensor`.
        input_tensor = tf.convert_to_tensor(image)
        # The model expects a batch of images, so add an axis with `tf.newaxis`.
        input_tensor = input_tensor[tf.newaxis,...]

        # Run inference
        output_dict = model(input_tensor)
        # All outputs are batches tensors.
        # Convert to numpy arrays, and take index [0] to remove the batch dimension.
        # We're only interested in the first num_detections.
        num_detections = int(output_dict.pop('num_detections'))
        output_dict = {key:value[0, :num_detections].numpy() 
                        for key,value in output_dict.items()}
        output_dict['num_detections'] = num_detections

        # detection_classes should be ints.
        output_dict['detection_classes'] = output_dict['detection_classes'].astype(np.int64)
        
        # Handle models with masks:
        if 'detection_masks' in output_dict:
            # Reframe the the bbox mask to the image size.
            detection_masks_reframed = utils_ops.reframe_box_masks_to_image_masks(
                    output_dict['detection_masks'], output_dict['detection_boxes'],
                    image.shape[0], image.shape[1])      
            detection_masks_reframed = tf.cast(detection_masks_reframed > 0.5,
                                            tf.uint8)
            output_dict['detection_masks_reframed'] = detection_masks_reframed.numpy()
        
        return output_dict
