
import numpy as np
import os
import six.moves.urllib as urllib
import sys
import time
import tensorflow as tf

from collections import defaultdict
from io import StringIO
from matplotlib import pyplot as plt
from PIL import Image, ImageDraw
import cv2

from object_detection.utils import ops as utils_ops
from object_detection.utils import label_map_util
from object_detection.utils import visualization_utils as vis_util

# patch tf1 into `utils.ops`
utils_ops.tf = tf.compat.v1

# Patch the location of gfile
tf.gfile = tf.io.gfile

# Enable Eager Execution
tf.compat.v1.enable_eager_execution()

print("Start")
PATH_TO_LABELS = os.path.join(os.path.expanduser("~"), "Projects", "Tensorflow",'models/research/object_detection/data/mscoco_label_map.pbtxt')
category_index = label_map_util.create_category_index_from_labelmap(PATH_TO_LABELS, use_display_name=True)


def run_inference_for_single_image(model, image):
  
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



video_capture = cv2.VideoCapture(0)
# take 10 images
ret, frame = video_capture.read()
time.sleep(0.5)

width, height = frame.shape[:2]
size = ( height, width)
model_id = "ssd_mobilenet_v2_coco_2018_03_29"
model_dir = os.path.join(os.path.expanduser("~"), "Projects", "Tensorflow", "Models", model_id, "saved_model")
print(model_dir)
model = tf.compat.v2.saved_model.load(model_dir, None)
model = model.signatures['serving_default']
print("Model Loaded")


fps = 0
count = 0
start = time.time()


while True:

  ret, frame = video_capture.read()
  frame = cv2.flip(frame, 0)

  width, height = frame.shape[:2]
  size = (width, height)
  output_dict = run_inference_for_single_image(model, frame)
  print([category_index[x]['name'] for x in output_dict['detection_classes']])
  for obj in output_dict['detection_boxes']:
    print(obj)
    bmin = (int(obj[1]*size[0]), int(obj[0]*size[1]))
    bmax = (int(obj[3]*size[0]), int(obj[2]*size[1]))
    cv2.rectangle(frame, bmin, bmax, (0, 255, 0), 2)
  cv2.imshow("Live", frame)


  end = time.time()
  if (end - start > 2):
      fps = int(count / 2)
      start = time.time()
      count = 0
      print(fps)
  else:
      count += 1


  if cv2.waitKey(10) & 0xFF == ord('q'):
    break

#cv2.imwrite('out.jpg', frame)
#print(output_dict)