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
    def __init__(self, model="ssd_mobilenet_v2_coco_2018_03_29", modelType="savedmodel"):
        modelDir = os.path.join('Traindata', 'model', model )
        print(f'Load Model from: {modelDir}')
        self.modelType = modelType
        tf.compat.v1.enable_eager_execution

        if self.modelType == "tflite":
            model = ''
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
            model = os.path.join(modelDir, "saved_model" )
            self.model = tf.compat.v2.saved_model.load(model, None)
            self.model = self.model.signatures['serving_default']
        elif self.modelType == "graph":
            model = ''
            for f in os.listdir(modelDir):
                if f.endswith('.pb'):
                    model = os.path.join(modelDir, f)
                    break
            print(model)
            detection_graph = tf.Graph()
            with detection_graph.as_default():
                od_graph_def = tf.GraphDef()
                with tf.gfile.GFile(model, 'rb') as fid:
                    serialized_graph = fid.read()
                    od_graph_def.ParseFromString(serialized_graph)
                    tf.import_graph_def(od_graph_def, name='')
            self.model = detection_graph
            

    def prepareImage(self, image, scale):
        logging.debug("Prepare Image")
        width, height = image.shape[:2]
        self.image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        self.image = cv2.cv2.resize(image, (int(height / scale), int(width / scale)))

        return self.image

    def detectFace(self, image=None, normalized=True):
        if not image:
            image = self.image
        #logging.debug(image)
        if self.modelType == "tflite":
            self.detections = self.run_inference_on_tflite(image)
            pass
        elif self.modelType == "savedmodel":
            self.detections = self.run_inference_on_saved_model(self.model, image)
        elif self.modelType == "graph":
            self.detections = self.run_inference_on_graph(self.model, image)
        return self.detections


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

    def run_inference_on_graph(self, graph, image):
        with graph.as_default():
            with tf.Session() as sess:
                print("Prepare Model")
                # Get handles to input and output tensors
                ops = tf.get_default_graph().get_operations()
                all_tensor_names = {output.name for op in ops for output in op.outputs}
                tensor_dict = {}
                for key in [
                    'num_detections', 'detection_boxes', 'detection_scores',
                    'detection_classes', 'detection_masks'
                ]:
                    tensor_name = key + ':0'
                    if tensor_name in all_tensor_names:
                        tensor_dict[key] = tf.get_default_graph().get_tensor_by_name(
                            tensor_name)
                if 'detection_masks' in tensor_dict:
                    # The following processing is only for single image
                    detection_boxes = tf.squeeze(tensor_dict['detection_boxes'], [0])
                    detection_masks = tf.squeeze(tensor_dict['detection_masks'], [0])
                    # Reframe is required to translate mask from box coordinates to image coordinates and fit the image size.
                    real_num_detection = tf.cast(tensor_dict['num_detections'][0], tf.int32)
                    detection_boxes = tf.slice(detection_boxes, [0, 0], [real_num_detection, -1])
                    detection_masks = tf.slice(detection_masks, [0, 0, 0], [real_num_detection, -1, -1])
                    detection_masks_reframed = utils_ops.reframe_box_masks_to_image_masks(
                        detection_masks, detection_boxes, image.shape[0], image.shape[1])
                    detection_masks_reframed = tf.cast(
                        tf.greater(detection_masks_reframed, 0.5), tf.uint8)
                    # Follow the convention by adding back the batch dimension
                    tensor_dict['detection_masks'] = tf.expand_dims(
                        detection_masks_reframed, 0)
                image_tensor = tf.get_default_graph().get_tensor_by_name('image_tensor:0')

                print("Run inference")
                output_dict = sess.run(tensor_dict,
                                        feed_dict={image_tensor: np.expand_dims(image, 0)})
                
                # all outputs are float32 numpy arrays, so convert types as appropriate
                output_dict['num_detections'] = int(output_dict['num_detections'][0])
                output_dict['detection_classes'] = output_dict[
                    'detection_classes'][0].astype(np.uint8)
                output_dict['detection_boxes'] = output_dict['detection_boxes'][0]
                output_dict['detection_scores'] = output_dict['detection_scores'][0]
                if 'detection_masks' in output_dict:
                    output_dict['detection_masks'] = output_dict['detection_masks'][0]
               
        return output_dict


    def run_inference_on_saved_model(self, model, image):
    
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
        num_detections = output_dict.pop('num_detections')
        print(num_detections.numpy())
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

    def run_inference_on_tflite(self, img):

        img_org = img.copy()

        #prepare Image
        img = cv2.resize(img, (300, 300))
        img = img.reshape(1, img.shape[0], img.shape[1], img.shape[2]) # (1, 300, 300, 3)
        img = img.astype(np.float32)


        # set input tensor
        self.interpreter.set_tensor(self.input_details[0]['index'], img)


        # run
        self.interpreter.invoke()
        output_dict = {}
        # get outpu tensor
        output_dict['detection_boxes'] = self.interpreter.get_tensor(self.output_details[0]['index'])
        output_dict['detection_classes'] = self.interpreter.get_tensor(self.output_details[1]['index'])
        output_dict['detection_scores'] = self.interpreter.get_tensor(self.output_details[2]['index'])
        output_dict['num_detections'] = int(self.interpreter.get_tensor(self.output_details[3]['index']))

        return output_dict
