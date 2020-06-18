
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

#Model
model_id = "ssd_mobilenet_v2_coco_2018_03_29"
model_dir = os.path.join(os.path.expanduser("~"), "Projects", "Tensorflow", "Models", model_id, "saved_model")


video_capture = cv2.VideoCapture(0)
ret, frame = video_capture.read()
time.sleep(0.5)
width, height = frame.shape[:2]
size = ( height, width)

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