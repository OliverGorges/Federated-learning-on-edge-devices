
import numpy as np
import os
import six.moves.urllib as urllib
import sys
import time
import tensorflow as tf
from google.protobuf import text_format
from collections import defaultdict
from io import StringIO
from matplotlib import pyplot as plt
from PIL import Image, ImageDraw
import cv2

from object_detection.utils import ops as utils_ops
from object_detection.utils import label_map_util
from object_detection.utils import visualization_utils as vis_util


from utils.ThermalImage.camera import Camera as ThermalCamera
from utils.ThermalImage.postprocess import dataToImage

# patch tf1 into `utils.ops`
utils_ops.tf = tf.compat.v1

# Patch the location of gfile
tf.gfile = tf.io.gfile

# Enable Eager Execution
tf.compat.v1.enable_eager_execution()

print("Start")
PATH_TO_LABELS = os.path.join("Traindata", "data", "labelmap.pbtxt")
category_index = label_map_util.create_category_index_from_labelmap(PATH_TO_LABELS, use_display_name=True)

#Model
model_id = "FaceDetect"
model_dir = os.path.join("Traindata", "model", model_id)


#video_capture = cv2.VideoCapture(0)
#ret, frame = video_capture.read()
tc = ThermalCamera(uid="LcN")
time.sleep(0.5)

print(model_dir)
#model = tf.compat.v2.saved_model.load(model_dir, None)
#model = model.signatures['serving_default']
detection_graph = tf.Graph()
with detection_graph.as_default():
  od_graph_def = tf.GraphDef()
  with tf.gfile.GFile(os.path.join(model_dir, "frozen_inference_graph.pb"), 'rb') as fid:
    serialized_graph = fid.read()
    od_graph_def.ParseFromString(serialized_graph)
    tf.import_graph_def(od_graph_def, name='')
print("Model Loaded")


fps = 0
count = 0
start = time.time()



def run_inference_for_single_image(image, graph):
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




while True:
  print("Take Image")
  #ret, frame = video_capture.read()
  #frame = cv2.flip(frame, 0)
  data = tc.getTemperatureImage()
  frame = dataToImage(data, (640,480))

  width, height = frame.shape[:2]
  size = (width, height)
  output_dict = run_inference_for_single_image(frame, detection_graph)
  print([category_index[x]['name'] for x in output_dict['detection_classes']])
  for obj in output_dict['detection_boxes']:
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