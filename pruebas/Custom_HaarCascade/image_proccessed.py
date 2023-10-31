from __future__ import annotations

import os

import cv2 as cv
import numpy as np
import tqdm
for imgpath in tqdm.tqdm(os.listdir('data/original')):
    img = cv.imread('data/original/' + imgpath, cv.IMREAD_GRAYSCALE)
    resized_image = cv.resize(img, (24, 24))
    normalizedImg = np.zeros((24, 24))
    normalizedImg = cv.normalize(
        resized_image,  normalizedImg, 0, 255, cv.NORM_MINMAX,
    )
    cv.imwrite('data/processed/' + imgpath, resized_image)
