import cv2
import time
import numpy as np
from PIL import ImageGrab
from matplotlib import pyplot as plt
from pymouse import PyMouse

m = PyMouse()

template = cv2.imread('balloon.PNG',0)
w, h = template.shape[::-1]

methods = ['cv2.TM_CCOEFF','cv2.TM_CCOEFF_NORMED','cv2.TM_CCORR',
            'cv2.TM_CCORR_NORMED', 'cv2.TM_SQDIFF', 'cv2.TM_SQDIFF_NORMED']

def multiFind():
    capture = ImageGrab.grab()
    capture = np.array(capture)
    cv_capture = capture.astype(np.uint8)
    cv_capture_grey = cv2.cvtColor(cv_capture, cv2.COLOR_BGR2GRAY)

    res = cv2.matchTemplate(cv_capture_grey,template,cv2.TM_CCOEFF_NORMED)
    threshold = 0.75
    loc = np.where( res >= threshold)
    print(len(zip(*loc[::-1])))
    for pt in zip(*loc[::-1]):
        m.move(pt[0]+(w/2), pt[1]+(h/2))
        m.click( pt[0]+(w/2), pt[1]+(h/2))

while True:
    multiFind()
    time.sleep(1)
