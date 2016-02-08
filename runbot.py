import cv2
import numpy as np
from PIL import ImageGrab
from matplotlib import pyplot as plt

from pymouse import PyMouse
m = PyMouse()

capture = ImageGrab.grab()
capture = np.array(capture)
cv_capture = capture.astype(np.uint8)
cv_capture_grey = cv2.cvtColor(cv_capture, cv2.COLOR_BGR2GRAY)

img = cv_capture_grey
img2 = img.copy()
template = cv2.imread('balloon.PNG',0)
w, h = template.shape[::-1]

methods = ['cv2.TM_CCOEFF','cv2.TM_CCOEFF_NORMED','cv2.TM_CCORR',
            'cv2.TM_CCORR_NORMED', 'cv2.TM_SQDIFF', 'cv2.TM_SQDIFF_NORMED']

def testMatching():
    for meth in methods:
        img = img2.copy()
        method = eval(meth)


        #apply template match
        res = cv2.matchTemplate(img,template,method)
        print(res)
        min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)

        #if method is TM_SQDIFF or TM_SQDIFF_NORMED use minimum
        if method in [cv2.TM_SQDIFF_NORMED, cv2.TM_SQDIFF]:
            top_left = min_loc
        else:
            top_left = max_loc
        bottom_right = (top_left[0] + w, top_left[1] + h)

        cv2.rectangle(img,top_left,bottom_right,255,2)

        plt.subplot(121),plt.imshow(res,cmap='magma')
        plt.title('matching result'), plt.xticks([]), plt.yticks([])
        plt.subplot(122), plt.imshow(img,cmap='magma')
        plt.title('detected point'), plt.xticks([]), plt.yticks([])
        plt.suptitle(meth)

        plt.show()

def multiFind():
    res = cv2.matchTemplate(cv_capture_grey,template,cv2.TM_CCOEFF_NORMED)
    threshold = 0.50
    loc = np.where( res >= threshold)
    for pt in zip(*loc[::-1]):
        cv2.rectangle(cv_capture_grey, pt, (pt[0] + w, pt[1] + h), (0,0,255),2)
        m.click( pt[0]+(w/2), pt[1]+(h/2))

multiFind()
cv2.imwrite("result.png",cv_capture_grey)
