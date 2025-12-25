import cv2
import numpy as np
import sys

print("Python:", sys.version.split()[0])
print("OpenCV:", cv2.__version__)

# 生成一张简单图像并检测 ORB 特征，保存可视化结果
img = np.zeros((300, 300), dtype='uint8')
cv2.rectangle(img, (60, 60), (240, 240), 255, -1)

orb = cv2.ORB_create(nfeatures=200)
kp, des = orb.detectAndCompute(img, None)
print("Keypoints detected:", len(kp))

vis = cv2.drawKeypoints(img, kp, None, color=(255,0,0))
cv2.imwrite('test_block.jpg', img)
cv2.imwrite('test_orb_kp.jpg', vis)
print("Wrote test_block.jpg and test_orb_kp.jpg")
