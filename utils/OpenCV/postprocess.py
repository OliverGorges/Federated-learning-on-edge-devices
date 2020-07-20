import cv2
import logging

"""
Methods that a used to display the Images
"""

def drawBoxes(image, boxes, color=(0, 255, 0)):
    width, height = image.shape[:2]
    for (x, y, w, h) in boxes:
        x = int(x * width)
        y = int(y * height)
        w = int(w * width)
        h = int(h * height)
        logging.debug((x, y, w, h))
        cv2.rectangle(image, (x, y), (w, h), color , 2)
    return image

