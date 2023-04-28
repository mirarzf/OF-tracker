import argparse
import logging
from pathlib import Path
from tqdm import tqdm

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

from unet.unetutils.data_loading import MasterDataset
from unet.unet_model import UNet, UNetAtt
from unet.unetutils.utils import plot_img_and_mask
from test import mask_to_image

import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'


# CHOOSE INPUT DIRECTORIES 
## RGB input 
# imgdir = Path('D:\\Master Thesis\\data\\KU\\train')
# imgdir = Path('D:\\Master Thesis\\data\\KU\\test')
imgdir = Path("./data/test/imgs-randomsplit")
imgfilenames = [f for f in imgdir.glob('*.png') if f.is_file()] 

## Attention maps input 
attmapdir = None # None when you don't want attention maps 
# attmapdir = Path('D:\\Master Thesis\\data\\KU\\testannot')
# attmapdir = Path("./data/test/attmaps")

## Optical Flow input 
# flowdir = None # None when you don't want optical flow 
flowdir = Path("./data/test/flows")

## Folder where to save the predicted segmentation masks 
outdir = Path("./results/unet")

## Checkpoint directories 
dir_checkpoint = Path('./checkpoints')
### Model file path inside dir_checkpoint folder 
ckp = "U-Net-2-no-rgb-w-flow/checkpoint_epoch_best.pth"


def predict_img(net,
                full_img: Image,
                out_filename: str, 
                device,
                full_attmap: Image = None, 
                full_flow: np.ndarray = None, 
                img_scale: float = 1,
                mask_threshold: float =0.5, 
                useatt: bool = False, 
                useflow: bool = False, 
                addpositions: bool = False, 
                rgbtogs: bool = False, 
                noimg: bool = False, 
                savepred: bool = True, 
                visualize: bool = False):
    net.eval()
    imgsize = full_img.size
    img = torch.from_numpy(MasterDataset.preprocess(full_img, img_scale, is_mask=False))
    
    # In the case of no rgb input 
    lastimgchannel = 3
    if rgbtogs: 
        lastimgchannel = 1

    # Add optical flow input if toggled on 
    if useflow: 
        flow = MasterDataset.normalizeflow(full_flow)
        flow = flow.transpose((2, 0, 1)) # Transpose dimensions of optical flow array so that they become (2, width, height) 
        flo_tensor = torch.as_tensor(flow.copy()).float().contiguous() # Transform into a tensor beforeapplying interpolation to change input size 
        # Then change the size of your tensor before adding it to the input dictionary 
        flo_tensor = flo_tensor.unsqueeze(0) 
        flo_tensor = torch.nn.functional.interpolate(input=flo_tensor, size=(300,300), mode='bicubic', align_corners=True)
        flo_tensor = flo_tensor.squeeze()
        # Concatenate to image 
        img = torch.cat((img, flo_tensor), dim=0)

    # Add normalized positions to input if toggled on 
    if addpositions: 
        _, w, h = img.shape
        x = torch.tensor(np.arange(h)/(h-1))
        y = torch.tensor(np.arange(w)/(w-1))
        grid_x, grid_y = torch.meshgrid(x, y, indexing='ij')
        grid_x = grid_x.repeat(1, 1, 1)
        grid_y = grid_y.repeat(1, 1, 1)
        img = torch.cat((img, grid_x, grid_y), dim=0)
        
    # remove RGB image or Grayscale image input if toggled on 
    if noimg: 
        img = img[lastimgchannel:,:,:]

    img = img.unsqueeze(0)
    img = img.to(device=device, dtype=torch.float32)

    if useatt: 
        attmap = torch.from_numpy(MasterDataset.preprocess(full_attmap, img_scale, is_mask=True))
        attmap = attmap.unsqueeze(0)
        attmap = attmap.to(device=device, dtype=torch.float32)

    with torch.no_grad():
        # predict the mask 
        if useatt: 
            mask_pred = net(img, attmap)
        else: 
            mask_pred = net(img)
            
        if net.n_classes == 1:
            # convert to one-hot format
            mask_pred = (F.sigmoid(mask_pred) > mask_threshold).float()
        else:
            mask_pred = mask_pred.argmax(dim=1)
        
        mask_pred_img = mask_to_image(mask_pred[0].cpu().numpy(), net.n_classes, imgsize)
        if savepred: 
            logging.info(f"Mask saved to {out_filename}")
            mask_pred_img.save(out_filename)
        if visualize:
            plot_img_and_mask(img[0].cpu().numpy().transpose((1,2,0))[:,:,:3], mask_pred[0].cpu().numpy(), net.n_classes) 
            logging.info(f'Visualizing results for image {filename}, close to continue...')

def get_args():
    parser = argparse.ArgumentParser(description='Predict masks from input images')
    parser.add_argument('--model', '-m', default=dir_checkpoint / ckp, metavar='FILE',
                        help='Specify the file in which the model is stored')
    inputgroup = parser.add_mutually_exclusive_group(required=True)
    inputgroup.add_argument('--input', '-i', metavar='INPUT', nargs='+', help='Filenames of input images')
    parser.add_argument('--input_att', '-a', metavar='INPUT ATTENTION', nargs='+', help='Filenames of input attention maps')
    parser.add_argument('--input_flow', '-of', metavar='INPUT OPTICAL FLOW', nargs='+', help='Filenames of input optical flows')
    inputgroup.add_argument('--dir', action='store_true', default=False, help='Use directories specified in predict.py file instead')
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
    parser.add_argument('--pos', action='store_true', default=False, help='Add normalized position to input')
    parser.add_argument('--grayscale', '-gs', action='store_true', default=False, help='Convert RGB image to Greyscale for input')
    parser.add_argument('--noimg', action='store_true', default=False, help='No image as input')

    return parser.parse_args()


def get_output_filenames(args): 
    return args.output or [outdir / f.name for f in imgfilenames]
    
def get_attmap_filenames(args): 
    if args.dir and attmapdir != None: 
        return [attmapdir / f.name for f in imgfilenames]
    else: 
        return args.input_att 
    
def get_flow_filenames(args): 
    if args.dir and flowdir != None: 
        return [flowdir / (f.stem + '.npy') for f in imgfilenames]
    else: 
        return args.input_flow 

if __name__ == '__main__':
    args = get_args()
    logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')

    # Prepare the input files 
    if args.dir: 
        in_files = imgfilenames
    else: 
        in_files = args.input
    in_files_att = get_attmap_filenames(args)
    in_files_flow = get_flow_filenames(args)
    out_files = get_output_filenames(args)

    useatt = True if in_files_att != None else False 
    useflow = True if in_files_flow != None else False 

    n_channels = 3 
    if args.grayscale: 
        n_channels = 1 
    elif args.noimg: 
        n_channels = 0 
    if args.pos: 
        n_channels += 2 
    if useflow: 
        n_channels += 2 
    if useatt: 
        net = UNetAtt(n_channels=n_channels, n_classes=args.classes, bilinear=args.bilinear)
    else: 
        net = UNet(n_channels=n_channels, n_classes=args.classes, bilinear=args.bilinear)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logging.info(f'Loading model {args.model}')
    logging.info(f'Using device {device}')

    modelToLoad = torch.load(args.model, map_location=device)
    nchanToLoad = modelToLoad['inc.double_conv.0.weight'].shape[1]
    assert net.n_channels == nchanToLoad, \
        f"Number of input channels ({net.n_channels}) and loaded model ({nchanToLoad}) are not the same. Choose a different model to load."
    net.load_state_dict(modelToLoad, strict=False)
    net.to(device=device)

    logging.info('Model loaded!')

    for i, filename in enumerate(tqdm(in_files)):
        logging.info(f'\nPredicting image {filename} ...')
        img = Image.open(filename)
        if useatt: 
            attmap = Image.open(in_files_att[i])
        else: 
            attmap = None 
        if useflow: 
            flow = np.load(in_files_flow[i])
        else: 
            flow = None 
        
        out_filename = out_files[i]
        predict_img(net=net,
                    full_img=img,
                    out_filename=out_filename, 
                    device=device, 
                    full_attmap=attmap, 
                    full_flow=flow, 
                    img_scale=args.scale,
                    mask_threshold=args.mask_threshold,
                    useatt=useatt, 
                    useflow=useflow, 
                    rgbtogs=args.grayscale, 
                    noimg=args.noimg, 
                    addpositions=args.pos, 
                    savepred=not args.no_save, 
                    visualize=args.viz)