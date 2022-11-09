import numpy as np 
import cv2 as cv 

from torch.utils.data import Dataset, DataLoader

import torch 
import torch.nn as nn 

class HandDataset(Dataset):
    def __init__(self, dataset):
        super(HandDataset, self).__init__()
    
    def __len__(self): 
        return None 

    def __getitem__(self, index): 
        return {}

def prob2seg(prob):
    """
    convert probability map to 0/255 segmentation mask
    prob: probability map [0-255]
    """
    # smooth and thresholding
    prob = cv.GaussianBlur(prob, (5, 5), 0)
    ret, mask = cv.threshold(prob,75,1,cv.THRESH_BINARY) # would remove the single-channel dimension
    
    # remove holes and spots
    kernel = np.ones((5,5),np.uint8)
    mask_open = cv.morphologyEx(mask, cv.MORPH_OPEN, kernel)
    mask_close = cv.morphologyEx(mask_open, cv.MORPH_CLOSE, kernel)
    
    # filter out small area
    contours, hierarchy = cv.findContours(mask_close, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_NONE)
    min_ratio = 0.002
    max_ratio = 0.2
    area_img = prob.shape[0] * prob.shape[1]
    mask_close = mask_close * 0
    for i in range(len(contours)):
        area = cv.contourArea(contours[i])
        if area > area_img*min_ratio and area < area_img*max_ratio:
            cv.drawContours(mask_close, [contours[i]], -1, 1, -1)

    return mask_close * 255