#encoding: utf-8
import cv2                   # 运行时要加sudo
import numpy as np
img = cv2.imread("load_data/input.bmp")  # 读取一张图像
print(img.shape)
img_100x96 = cv2.resize(img, (320, 375))
cv2.imwrite('load_data/input.bmp', img_100x96)
