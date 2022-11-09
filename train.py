import os 
import argparse

import numpy as np 
import cv2 as cv 

import torch 
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import torch.backends.cudnn as cudnn

from tqdm import tqdm

## DEFINE FOLDERS 
outputfolder = "/results"

def train(args): 
    ## RETRIEVING DATA 

    ## INITIALIZING MODEL WITH SEEDED RANDOM WEIGHTS 

    ## SETTING UP PARAMETERS FOR MODEL AND TRAINING 
    # optimizer 
    # loss 
    # learning rate 
    lr = 1e-4
    # nb of epochs 
    nbEpochs = 10 
    # now put the mnodel in train mode 

    ## START EPOCHS 
    for epoch in tqdm(range(0, nbEpochs)): 
        print("training epoch numer :", epoch)

    return None 

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="This script creates a video containing all the resulting map of attention "
                                                "to focus the segmentation map around the hand")
    parser.add_argument('--model', default='raft-things.pth', help="restore checkpoint")
    parser.add_argument('--dataset', default='C:\\Users\\hvrl\\Documents\\data\\KU\\centerpoints.csv', help="CSV file with annotated points")
    parser.add_argument('--videofolder', '-vf', default='C:\\Users\\hvrl\\Documents\\data\\KU\\videos', help="folder containig the annotated videos")
    parser.add_argument('--scale', default=0.5, type=float, help="scale to resize the video frames. Default: 0.5")
    parser.add_argument('--small', action='store_true', help='use small model')
    parser.add_argument('--mixed_precision', action='store_true', help='use mixed precision')
    parser.add_argument('--alternate_corr', action='store_true', help='use efficent correlation implementation')
    args = parser.parse_args()

    train(args)