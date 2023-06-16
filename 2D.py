import cv2
import numpy as np

img = cv2.imread('xiaarea.png')
height,width,channel = img.shape

def translate(image, x, y):

    M = np.float32([[1, 0, x], [0, 1, y]])
    move = cv2.warpAffine(image, M, (image.shape[1], image.shape[0]))
    move
    return shifted

def rotate(image, angle, center = None, scale = 1.0):
    (h, w) = image.shape[:2]

    if center is None:
        center = (w / 2, h / 2)

    M = cv2.getRotationMatrix2D(center, angle, scale)
    rotat = cv2.warpAffine(image, M, (w, h))

    return rotat

def suo_fang(img, fx,fy):
    M = np.float32([[fx, 0, 0], [0, fy, 0]])
    resized = cv2.warpAffine(img, M, (int(width*fx), int(height*fy)))

    return resized

def fan_zhuan(img,t ):
    if (t == "level"):
        M = np.float32([[-1, 0, width], [0, 1, 0]])
    elif(t == "vertical"):
        M = np.float32([[1, 0, 0], [0, -1, height]])
    elif(t == "level_vertical"):
        M = np.float32([[-1, 0, width], [0, -1, height]])
    flip =  cv2.warpAffine(img, M, (width, height))
    return flip


shifted = translate(img, 10, 30)
rotates = rotate(img,90)
resized = suo_fang(img,3,1)
for i in ["level","vertical","level_vertical"]:
    flip = fan_zhuan(img ,i)
    cv2.imwrite("flip_"+i+".jpg",flip)


cv2.imwrite('resize_raw.jpg', resized)
cv2.imwrite('shift_right_10_down_30.png', shifted)
cv2.imwrite('rotates_90_1.0.png', rotates)