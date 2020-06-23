import cv2
from copy import deepcopy
import numpy as np

def dataToImage(data, outputSize):
    img = np.asarray(deepcopy(data))
    img = img.reshape(60, 80)
    thermalImg = cv2.normalize(img, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
    thermalImg = cv2.flip(thermalImg, 0)
    thermalImg =cv2.cv2.resize(thermalImg, (outputSize))
    thermalImg = cv2.applyColorMap(thermalImg, cv2.COLORMAP_HOT)
    return thermalImg


