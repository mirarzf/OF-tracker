import argparse
import logging
import os
from pathlib import Path
from tqdm import tqdm


import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
import cv2 as cv 
import matplotlib.pyplot as plt 
from torchvision import transforms

from unet.unetutils.data_loading import MaskDataset
from torch.utils.data import DataLoader
from unet.unet_model import UNet, UNetAtt

# Directories 
dir_img = Path('D:\\Master Thesis\\data\\KU\\test')

dir_mask = Path('D:\\Master Thesis\\data\\KU\\testannot')

dir_attmap = Path('./data/attmaps/')

outdir = Path("./results/unet/celoss")
outdirpred = Path("./results/unet/pred")

dir_model = "./checkpoints/"
defaultmodelname = "woattention/tKU_bs4_e10_new.pth"
modelpath = Path(dir_model + defaultmodelname)

def test_img(net,
            device,
            img_scale: float = 0.5,
            useatt: bool = False, 
            modelname = Path(modelpath).stem):
    
    # 1. Create dataset
    if useatt: 
        dataset = MaskDataset(images_dir=dir_img, masks_dir=dir_mask, scale=img_scale, attmaps_dir=dir_attmap)
    else: 
        dataset = MaskDataset(images_dir=dir_img, masks_dir=dir_mask, scale=img_scale, attmaps_dir='', withatt=False)

    # 2. Create data loaders
    loader_args = dict(batch_size=1, num_workers=4, pin_memory=True)
    data_loader = DataLoader(dataset, shuffle=True, **loader_args)

    # 3. Set the loss 
    criterion = nn.CrossEntropyLoss(reduction='none') # VANILLA 
    
    # 4. Create results saving folder 
    outputfolder = outdir / modelname
    Path.mkdir(outputfolder, exist_ok=True)
    outputfolderpred = outdirpred / modelname
    Path.mkdir(outputfolderpred, exist_ok=True)

    # 5. Begin prediction 
    for batch in data_loader: 
        images = batch['image']
        true_masks = batch['mask']
        indices = batch['index']
        filenames = batch['filename']

        if useatt: 
            attention_maps = batch['attmap']
        
        assert images.shape[1] == net.n_channels, \
            f'Network has been defined with {net.n_channels} input channels, ' \
            f'but loaded images have {images.shape[1]} channels. Please check that ' \
            'the images are loaded correctly.'
        
        images = images.to(device=device, dtype=torch.float32)
        true_masks = true_masks.to(device=device, dtype=torch.long)
        indices = indices.to(device=device, dtype=torch.int)
        if useatt: 
            attention_maps = attention_maps.to(device=device, dtype=torch.float32)


        with torch.no_grad():
            masks_pred = net(images)
            if net.n_classes == 1:
                loss = criterion(masks_pred.squeeze(1), true_masks.float())
                masks_pred = F.softmax(masks_pred, dim=1)[0]
            else:
                loss = criterion(masks_pred, true_masks)
                masks_pred = torch.sigmoid(masks_pred)[0]

        # 6. Save the loss image 
        tf = transforms.Compose([
            transforms.Normalize(torch.min(loss),torch.max(loss)), 
            transforms.ToPILImage()
        ])
        # tfpred = transforms.Compose([
        #     transforms.ToPILImage(),
        #     transforms.ToTensor()
        # ])
        
        loss_array = tf(loss.cpu())
        full_mask = masks_pred.cpu().squeeze()
        
        if net.n_classes == 1:
            full_mask = full_mask.float()
        else:
            print(full_mask.size())
            full_mask = 1-F.one_hot(full_mask.argmax(dim=0), net.n_classes).permute(2, 0, 1).float()
        tfnorm = transforms.Compose([
            transforms.ToPILImage()
        ])
        
        masks_pred_array = tfnorm(full_mask)
        # print(loss_array.getextrema(),masks_pred_array.getextrema())
        # loss_array.show()
        outfile = outputfolder / Path(filenames[0] + ".png")
        loss_array = loss_array.save(outfile)
        outfilepred = outputfolderpred / Path(filenames[0] + ".png")
        masks_pred_array = masks_pred_array.save(outfilepred)
        # print("Reach end of step 6")


    print("reach end of test-img function")
    return 0 

def get_args():
    parser = argparse.ArgumentParser(description='Predict masks from input images')
    parser.add_argument('--load', '-f', type=str, default=modelpath, help='Load model from a .pth file')
    parser.add_argument('--scale', '-s', type=float, default=1.0, help='Downscaling factor of the images')
    parser.add_argument('--bilinear', action='store_true', default=False, help='Use bilinear upsampling')
    parser.add_argument('--classes', '-c', type=int, default=2, help='Number of classes')
    parser.add_argument('--attention', action='store_true', default=False, help='Use UNet with attention model')

    return parser.parse_args()

if __name__== '__main__': 
    args = get_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    if args.attention: 
        net = UNetAtt(n_channels=3, n_classes=args.classes, bilinear=args.bilinear)
    else: 
        net = UNet(n_channels=3, n_classes=args.classes, bilinear=args.bilinear)

    modelToLoad = torch.load(args.load, map_location=device)
    net.load_state_dict(modelToLoad, strict=False)
        
    net.to(device=device)
    
    test_img(net,
            device,
            img_scale = args.scale,
            useatt = args.attention, 
            modelname = Path(args.load).stem)

# python .\test.py -gt .\data\masks\gg4541_4629_extract_1.png -p .\data\masks\gg4541_4629_extract_1.png