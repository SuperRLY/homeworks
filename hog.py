
import os
import cv2
import numpy as np
import cv2
from skimage import feature
from skimage import exposure
img = cv2.imread('cqupt.jpg', 0)     #参数0表示读取灰度图片
cv2.imshow('IMREAD_gray', img)
img2 = np.power(img/float(np.max(img)), 1.5)    #幂运算，参数一为底数，参数二为指数，这是伽马矫正，使得图像更接近人眼所见
gx = cv2.Sobel(img2, cv2.CV_64F, 1, 0, ksize=1)   #ksize为Sobel算子大小，3是矩阵大小为3,0表示在x方向求梯度，1表示在y方向求梯度
gy = cv2.Sobel(img2, cv2.CV_64F, 0, 1, ksize=1)
gx1 = np.abs(gx)    ##无符号梯度
gy1 = np.abs(gy)
cv2.imshow('IMREAD_gray2_gx', gx1)
cv2.imshow('IMREAD_gray2_gy', gy1)
cv2.waitKey(0)
cv2.destroyAllWindows()
cv2.imwrite('IMREAD_gray2_gx.jpg', gx1*255)
cv2.imwrite('IMREAD_gray2_gy.jpg', gy1*255)   #保存图片

fd, hog_image = feature.hog(img, orientations=9, pixels_per_cell=(16, 16), cells_per_block=(2, 2), visualize=True)
hog_image_rescaled = exposure.rescale_intensity(hog_image, in_range=(0, 10))
cv2.imshow('img', img)
cv2.imshow('hog', hog_image_rescaled)
cv2.waitKey(0)
cv2.destroyAllWindows()
cv2.imwrite('hog.jpg', hog_image_rescaled*255)









