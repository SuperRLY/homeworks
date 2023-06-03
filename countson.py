import cv2
import numpy as np
img=cv2.imread('cqupt.jpg',0)
img_sobel=cv2.Sobel(img,cv2.CV_64F,1,0,ksize=3)#ksize是Sobel算子大小，3是矩阵大小为3
img_sobel+=cv2.Sobel(img,cv2.CV_64F,0,1,ksize=3)
cv2.imwrite('img_sobel.jpg',img_sobel)
kernelx=np.array([[1,1,1],[0,0,0],[-1,-1,-1]],dtype=int)
kernely=np.array([[-1,0,1],[-1,0,1],[-1,0,1]],dtype=int)
img_prewittx=cv2.filter2D(img,-1,kernelx)
img_prewitty=cv2.filter2D(img,-1,kernely)
img_prewitt=img_prewittx+img_prewitty
cv2.imwrite('img_prewitt.jpg',img_prewitt)
img_scharr=cv2.Scharr(img,cv2.CV_64F,1,0)
img_scharr+=cv2.Scharr(img,cv2.CV_64F,0,1)
cv2.imwrite('img_scharr.jpg',img_scharr)
img_laplacian=cv2.Laplacian(img,cv2.CV_64F)
cv2.imwrite('img_laplacian.jpg',img_laplacian)
img_canny=cv2.Canny(img,100,200)
cv2.imwrite('img_canny.jpg',img_canny)
kernelx=np.array([[1,0],[0,-1]],dtype=int)
kernely=np.array([[0,1],[-1,0]],dtype=int)
img_robertsx=cv2.filter2D(img,-1,kernelx)
img_robertsy=cv2.filter2D(img,-1,kernely)
img_roberts=img_robertsx+img_robertsy
cv2.imwrite('img_roberts.jpg',img_roberts)



