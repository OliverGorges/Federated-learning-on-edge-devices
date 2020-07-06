"""
Test for TFLite model convertion and prediction
"""

"""
https://gist.github.com/iwatake2222/e4c48567b1013cf31de1cea36c4c061c

Working cli commands: 

#Prepare Model for TFlite
tflite_convert --graph_def_file Federated-learning-on-edge-devices/Traindata/model/TFlite/tflite_graph.pb --output_file=./detect.tflite --output_format=TFLITE --input_shapes=1,300,300,3 --input_arrays=normalized_input_image_tensor --output_arrays='TFLite_Detection_PostProcess','TFLite_Detection_PostProcess:1','TFLite_Detection_PostProcess:2','TFLite_Detection_PostProcess:3' --allow_custom_ops

#Convert to TFlite
python Tensorflow/models/research/object_detection/export_tflite_ssd_graph.py --pipeline_config_path ~/Projects/Federated-learning-on-edge-devices/Traindata/model/ssd_mobilenet_v2_coco_2018_03_29/custom_pipeline.config --trained_checkpoint_prefix ~/Projects/Federated-learning-on-edge-devices/Traindata/model/FaceDetect/model.ckpt-0 --output_directory ~/Projects/Federated-learning-on-edge-devices/Traindata/model/TFlite

"""



from utils.ThermalImage.camera import Camera as ThermalCamera
from utils.ThermalImage.postprocess import dataToImage
import tensorflow as tf
import os
import time
import cv2
import numpy as np

path = os.path.join("Traindata", "model", "TFlite", "detect.tflite")



def predictImage(img):

    img_org = img.copy()
    # load model
    interpreter = tf.lite.Interpreter(model_path=path)
    interpreter.allocate_tensors()
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()


    #prepare Image
    img = cv2.resize(img, (300, 300))
    img = img.reshape(1, img.shape[0], img.shape[1], img.shape[2]) # (1, 300, 300, 3)
    img = img.astype(np.float32)


    # set input tensor
    interpreter.set_tensor(input_details[0]['index'], img)


    # run
    interpreter.invoke()

    # get outpu tensor
    boxes = interpreter.get_tensor(output_details[0]['index'])
    labels = interpreter.get_tensor(output_details[1]['index'])
    scores = interpreter.get_tensor(output_details[2]['index'])
    num = interpreter.get_tensor(output_details[3]['index'])


    
    for i in range(boxes.shape[1]):
        if scores[0, i] > 0.5:
            box = boxes[0, i, :]
            x0 = int(box[1] * img_org.shape[1])
            y0 = int(box[0] * img_org.shape[0])
            x1 = int(box[3] * img_org.shape[1])
            y1 = int(box[2] * img_org.shape[0])
            box = box.astype(np.int)
            cv2.rectangle(img_org, (x0, y0), (x1, y1), (255, 0, 0), 2)
            cv2.rectangle(img_org, (x0, y0), (x0 + 100, y0 - 30), (255, 0, 0), -1)
            cv2.putText(img_org,
                    str(int(labels[0, i])),
                    (x0, y0),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1,
                    (255, 255, 255),
                    2)

    print(labels)
    #	cv2.imwrite('output.jpg', img_org)
    cv2.imshow('image', img_org)




if __name__ == "__main__":
    
    #video_capture = cv2.VideoCapture(0)
    #ret, frame = video_capture.read()
    
    tc = ThermalCamera(uid="LcN")
    time.sleep(0.5)

    while True:
        #ret, frame = video_capture.read()
        data = tc.getTemperatureImage()
        frame = dataToImage(data, (300,300))
        #frame = cv2.flip(frame, 0)
        predictImage(frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break