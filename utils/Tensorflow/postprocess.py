import cv2
import logging

"""
Methods to process modelreuslts and Images
"""

def drawBoxes(image, detection, color=(0, 255, 0)):
    width, height = image.shape[:2]
    print(len(detection['detection_boxes']))
    if 'detection_boxes' in detection:
        
        print(max(detection['detection_scores']))
        for i in range(len(detection['detection_boxes'])):
            score = detection['detection_scores'][i]
            if score > 0.25:
                try:
                    (x, y, w, h) = detection['detection_boxes'][i]
                    name = detection['detection_classes'][i]
                    x = int(x * width)
                    y = int(y * height)
                    w = int(w * width)
                    h = int(h * height)
                    
                    cv2.rectangle(image, (y, x), (h, w), color , 2)
                    logging.debug(f'Box: {(x, y, w, h)} Class: {name} Score: {score}')
                    #cv2.putText(image,str(name), (x,y))
                except: 
                    print(f"Error {detection['detection_boxes'][i]}")
    return image
