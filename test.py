import argparse
import logging
import os
from pathlib import Path
from tqdm import tqdm

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms

from unet.unetutils.data_loading import MaskDataset
from unet.unet_model import UNet, UNetAtt

# Directories 
dir_gt = Path('D:\\Master Thesis\\data\\KU\\test')

dir_mask = Path('D:\\Master Thesis\\data\\KU\\testannot')

dir_attmap = Path('./data/attmaps/')

outdir = Path("./results/unet")

def test_img(device,
            img_scale=1,
            out_threshold=0.5, 
            useatt: bool = False, 
            full_attmap = None):
    
    # 1. Create dataset
    if useatt: 
        dataset = MaskDataset(images_dir=dir_gt, masks_dir=dir_mask, scale=img_scale, attmaps_dir=dir_attmap)
    else: 
        dataset = MaskDataset(images_dir=dir_gt, masks_dir=dir_mask, scale=img_scale, withatt=False)

    # 2. Set up the dice loss 
    

    # 

def get_args():
    parser = argparse.ArgumentParser(description='Predict masks from input images')
    parser.add_argument('--model', '-m', default='MODEL.pth', metavar='FILE',
                        help='Specify the file in which the model is stored')
    parser.add_argument('--input', '-i', metavar='INPUT', nargs='+', help='Filenames of input images', required=True)
    parser.add_argument('--output', '-o', metavar='OUTPUT', nargs='+', help='Filenames of output images')
    parser.add_argument('--viz', '-v', action='store_true',
                        help='Visualize the images as they are processed')
    parser.add_argument('--no-save', '-n', action='store_true', help='Do not save the output masks')
    parser.add_argument('--mask-threshold', '-t', type=float, default=0.5,
                        help='Minimum probability value to consider a mask pixel white')
    parser.add_argument('--scale', '-s', type=float, default=0.5,
                        help='Scale factor for the input images')
    parser.add_argument('--bilinear', action='store_true', default=False, help='Use bilinear upsampling')
    parser.add_argument('--classes', '-c', type=int, default=2, help='Number of classes')
    parser.add_argument('--input_att', '-a', metavar='INPUT ATTENTION', nargs='+', help='Filenames of input attention maps')

    return parser.parse_args()