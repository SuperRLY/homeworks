import cv2
img = cv2.imread('cqupt.jpg', 0)     #参数0表示读取灰度图片
#先把img裁剪，旋转90度，再进行SIFT算法检测
img2 = img[0:200, 0:200]
img2 = cv2.transpose(img2)
img2 = cv2.flip(img2, 1)

sift = cv2.xfeatures2d.SIFT_create()
kp1, des1 = sift.detectAndCompute(img, None)
kp2, des2 = sift.detectAndCompute(img2, None)
#使用knnMatch匹配关键点，k=2，即对于每个关键点，返回两个最佳匹配
bf = cv2.BFMatcher()
matches = bf.knnMatch(des1, des2, k=2)
#提取匹配结果
good = []
for m, n in matches:
    if m.distance < 0.7*n.distance:
        good.append(m)
img3 = cv2.drawMatches(img, kp1, img2, kp2, good, None, flags=2)

cv2.imwrite('sift1.jpg', img3)



