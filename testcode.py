import cv2 as cv 
import numpy as np

import os 
from pathlib import Path

imgdir = Path("../Pytorch-UNet/results/probs")
framedir = Path("../data/KU/frames")
maskdir = Path("../data/KU/masks")

imgfilenames = [os.path.join(imgdir, f) for f in os.listdir(maskdir) if "mirrored" not in f]
framefilenames = [os.path.join(framedir, f) for f in os.listdir(maskdir) if "mirrored" not in f]
imgfilenames.sort()
framefilenames.sort()
print(imgfilenames)

n = len(imgfilenames)
for imgfilename, framefilename in zip(imgfilenames, framefilenames): 
    print(imgfilename)
    img = cv.imread(imgfilename)
    frame = cv.imread(framefilename)
    # seuil = np.quantile(img[:,:,0],0.5)
    # maxi = np.max(img)
    # mini = np.min(img)
    # print(maxi-mini)
    # img = (img - mini)/(maxi - mini)*255
    # img = np.uint8(img)
    # print("Le seuil est : ", seuil)
    # img = np.where(img >= seuil, 255, 0)
    print(img.shape, frame.shape)
    # print(np.count_nonzero(img[:,:,0] > 3))
    concat = cv.hconcat([img, frame])
    # cv.imshow("concat", concat)
    # cv.waitKey(0)
    cv.imwrite(os.path.join('results', os.path.basename(imgfilename)), concat)

print(img.shape[0]*img.shape[1])
print(img.shape)
