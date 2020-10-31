# https://opencv-python-tutroals.readthedocs.io/en/latest/py_tutorials/py_feature2d/py_sift_intro/py_sift_intro.html

import cv2
import numpy as np

img = cv2.imread('stop_sign.jpg')
gray= cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
# gray = cv2.cvtColor(np.float32(img), cv2.COLOR_RGB2GRAY)

sift = cv2.SIFT_create()

kp = sift.detect(gray,None)
print(kp)

# img_out = np.array('sift_keypoints.jpg')
# print(type(img_out))
# print(img_out)

img = cv2.drawKeypoints(gray,kp, img)

cv2.imwrite('sift_keypoints.jpg',img)

img=cv2.drawKeypoints(gray,kp, img, flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
cv2.imwrite('sift_keypoints.jpg',img)