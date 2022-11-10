import cv2 as cv 
import numpy as np 

image = cv.imread("D:\\Master Thesis\\data\\EgoHOS\\train\\label\\ego4d_000a3525-6c98-4650-aaab-be7d2c7b9402_600.png")

print(np.unique(image))