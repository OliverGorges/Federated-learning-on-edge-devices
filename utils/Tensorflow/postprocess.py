import cv2
import logging


def drawBoxes(image, detection, color=(0, 255, 0)):
    width, height = image.shape[:2]
    if 'num_detections' in detection:
        for i in range(int(detection['num_detections'])):
            (x, y, w, h) = detection['detection_boxes'][i]
            name = detection['detection_classes'][i]
            score = detection['detection_scores'][i]
            x = int(x * width)
            y = int(y * height)
            w = int(w * width)
            h = int(h * height)
            logging.debug(f'Box: {(x, y, w, h)} Class: {name} Score: {score}')
            cv2.rectangle(image, (x, y), (w, h), color , 2)
    return image
