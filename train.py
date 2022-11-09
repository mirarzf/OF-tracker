import sys 
sys.path.append('refinenetcore')
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

from modelcomplete import RefineNet, Bottleneck

from tqdm import tqdm

## DEFINE FOLDERS 
outputfolder = "/results"

def train(args): 
    ## PARSER ARGUMENTS SETTINGS 
    in_channels = args.in_channels 
    num_classes = args.num_classes 
    use_dropout = args.dropout 
    pretrained = True if (args.model != "") else False 

    ## RETRIEVING DATA 

    ## INITIALIZING MODEL WITH SEEDED RANDOM WEIGHTS 
    if torch.cuda.is_available(): 
        DEVICE = 'cuda'
    else: 
        DEVICE = 'cpu'
        
    model = RefineNet(Bottleneck, [3, 4, 23, 3], in_channels=in_channels, num_classes=num_classes, use_dropout=use_dropout, **kwargs)
    
    if pretrained: 
        trained_model = args.model 
        pretrained_dict = torch.load(trained_model)   
        model_dict = model.state_dict()

        pretrained_dict = {k: v for k,v in pretrained_dict.items() if k in model_dict and k.find('clf_conv')==-1 and k.find('conv1')==-1}
        model_dict.update(pretrained_dict)
        model.load_state_dict(model_dict)

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