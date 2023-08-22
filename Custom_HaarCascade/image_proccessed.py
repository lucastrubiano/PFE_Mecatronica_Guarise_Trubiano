import cv2 as cv
import numpy as np
import os
for imgpath in os.listdir('data/original'):
    img = cv.imread("data/original/" + '00001.png',cv.IMREAD_GRAYSCALE)
    resized_image = cv.resize(img, (24, 24))
    normalizedImg = np.zeros((24, 24))
    normalizedImg = cv.normalize(resized_image,  normalizedImg, 0, 255, cv.NORM_MINMAX)
    cv.imwrite("data/processed/" + imgpath, resized_image)