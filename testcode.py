import cv2 as cv 
import numpy as np

import os 
from pathlib import Path

path = "D:\\Master Thesis\\data\\EgoHOS\\train\\label\\ego4d_000a3525-6c98-4650-aaab-be7d2c7b9402_600.png"
print(path)
print(type(path))
img = cv.imread(path)
print(np.min(img), np.max(img))
imagtuned = np.where(img > 0, 255 / img, img)
print(np.unique(imagtuned), np.max(imagtuned))
cv.imshow('Imagtuned', imagtuned)
cv.waitKey(0)