import cv2
import numpy as np
from matplotlib import pyplot as plt

img = cv2.imread('screenshot.PNG',0)
img2 = img.copy()
template = cv2.imread('balloon.PNG',0)
w, h = template.shape[::-1]

methods = ['cv2.TM_CCOEFF','cv2.TM_CCOEFF_NORMED','cv2.TM_CCORR',
            'cv2.TM_CCORR_NORMED', 'cv2.TM_SQDIFF', 'cv2.TM_SQDIFF_NORMED']

for meth in methods:
    img = img2.copy()
    method = eval(meth)


    #apply template match
    res = cv2.matchTemplate(img,template,method)
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
