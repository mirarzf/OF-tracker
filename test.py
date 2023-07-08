import argparse
import logging
from pathlib import Path
from tqdm import tqdm

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

from torch.utils.data import DataLoader
from unet.unetutils.data_loading import MasterDataset, BasicDataset
from unet.unetutils.dice_score import multiclass_dice_coeff, dice_coeff
from unet.unet_model import UNet, UNetAtt
from unet.unetutils.utils import plot_img_and_mask_and_gt

import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'

# # REPRODUCIBILITY 
# import random
# def set_seed(seed: int = 42) -> None:
#     np.random.seed(seed)
#     random.seed(seed)
#     torch.manual_seed(seed)
#     torch.cuda.manual_seed(seed)
#     # When running on the CuDNN backend, two further options must be set
#     torch.backends.cudnn.deterministic = True
#     torch.backends.cudnn.benchmark = False
#     logging.info(f"Random seed set as {seed}. \n")
# # END REPRODUCIBILLITY 

# CHOOSE INPUT DIRECTORIES 
## RGB input 
# imgdir = Path('D:\\Master Thesis\\data\\KU\\train')
# imgdir = Path('D:\\Master Thesis\\data\\KU\\test')
imgdir = Path("./data/test/imgs")
imgdir = Path("./data/test/imgs-randomsplit")
imgfilenames = [f for f in imgdir.glob('*.png') if f.is_file()] 

## Ground truth masks 
# gtdir = Path('D:\\Master Thesis\\data\\KU\\trainannot')
# gtdir = Path('D:\\Master Thesis\\data\\KU\\testannot')
gtdir = Path("./data/test/masks")
gtdir = Path("./data/test/masks-randomsplit")

## Attention maps input 
attmapdir = None # None when you don't want attention maps 
# attmapdir = Path('D:\\Master Thesis\\data\\KU\\testannot')
# attmapdir = Path("./data/test/attmaps")

## Optical Flow input 
# flowdir = None # None when you don't want optical flow 
flowdir = Path("./data/test/flows")
flowdir = Path("./data/test/flows-randomsplit")

## Folder where to save the predicted segmentation masks 
outdir = Path("./results/unet")

## Checkpoint directories 
dir_checkpoint = Path('./checkpoints')
### Model file path inside dir_checkpoint folder 
ckp = "U-Net-3/checkpoint_epoch_best.pth"

##########################################################################################

def mask_to_image(mask: np.ndarray, n_classes = 2, finalsize = (0,0)):
    img = Image.fromarray((mask * 255 / (n_classes-1)).astype(np.uint8))
    if finalsize == (0,0): 
        finalsize = img.size
    return img.resize(finalsize)

def test_net(net, 
              device,
              images_dir, 
              masks_dir, 
              attmaps_dir, 
              flows_dir, 
              img_ids: list = [], 
              img_scale: float = 1.0,
              mask_threshold: float = 0.5, 
              useatt: bool = False, 
              useflow: bool = False, 
              addpositions: bool = False, 
              rgbtogs: bool = False, 
              noimg: bool = False, 
              savepred: bool = True, 
              visualize: bool = False):
    # 1. Create dataset
    if len(img_ids) == 0: 
        ids = [file.stem for file in images_dir.iterdir() if file.is_file() and file.name != '.gitkeep']
    else: 
        ids = img_ids 
    attmapdirForTest = '' if attmaps_dir == None else attmaps_dir
    flowdirForTest = '' if flows_dir == None else flows_dir
    test_set = MasterDataset(images_dir=images_dir, masks_dir=masks_dir, file_ids=ids, scale=img_scale, attmaps_dir=attmapdirForTest, withatt=useatt, flo_dir=flowdirForTest, withflo=useflow, grayscale=rgbtogs) 

    # 2. Create data loader 
    loader_args = dict(num_workers=4, pin_memory=True)
    test_loader = DataLoader(test_set, shuffle=False, batch_size=1, **loader_args)
    
    logging.info(f'''Starting testing:
        Attention model: {useatt}
        Positions input: {addpositions}
        Testing size:    {len(test_set)}
        Device:          {device.type}
        Images scaling:  {img_scale}
    ''')

    # 3. Calculate DICE score and save predicted masks if toggled on 
    net.eval()

    num_val_batches = len(test_loader)
    dice_score = 0
    
    # In the case of no rgb input 
    lastimgchannel = 3
    if rgbtogs: 
        lastimgchannel = 1

    # iterate over the test set
    for batch in tqdm(test_loader, total=num_val_batches, desc='Validation round', unit='batch', leave=False):
        image, mask_true = batch['image'], batch['mask']
        # add optical flow input if toggled on 
        if useflow: 
            opticalflow = batch['flow']
            image = torch.cat((image, opticalflow), dim=1)

        # add position input if toggled on  
        if addpositions: 
            # Add absolute positions to input 
            batchsize, _, w, h = image.shape
            x = torch.tensor(np.arange(h)/(h-1)) ################################ MULTIPLY BY 10 IN POSITIONS 
            y = torch.tensor(np.arange(w)/(w-1))
            grid_x, grid_y = torch.meshgrid(x, y, indexing='ij')
            grid_x = grid_x.repeat(len(image), 1, 1, 1)
            grid_y = grid_y.repeat(len(image), 1, 1, 1)
            image = torch.cat((image, grid_x, grid_y), dim=1)
        
        # remove image input if toggled on 
        if noimg: 
            image = image[:,lastimgchannel:,:,:]
        
        if useatt: 
            attmap = batch['attmap']
        # move images and labels to correct device and type
        image = image.to(device=device, dtype=torch.float32)
        mask_true = mask_true.to(device=device, dtype=torch.long)
        one_hot_mask_true = F.one_hot(mask_true, net.n_classes).permute(0, 3, 1, 2).float()
        
        # predict and compute DICE score 
        with torch.no_grad():
            # predict the mask
            if useatt: 
                mask_pred = net(image, attmap)
            else: 
                mask_pred = net(image)

            if net.n_classes == 1:
                # convert to one-hot format
                mask_pred = (F.sigmoid(mask_pred) > mask_threshold).float()
                # compute the Dice score
                dice_score += dice_coeff(mask_pred, one_hot_mask_true, reduce_batch_first=False)
            else:
                mask_pred = mask_pred.argmax(dim=1)
                # convert to one-hot format
                one_hot_mask_pred = F.one_hot(mask_pred, net.n_classes).permute(0, 3, 1, 2).float()
                # compute the Dice score, ignoring background (because multiclass)
                dice_score += multiclass_dice_coeff(one_hot_mask_pred[:, 1:, ...], one_hot_mask_true[:, 1:, ...], reduce_batch_first=False)

            # 4. Save or visualize predicted masks if toggled on 
            if savepred or visualize: 
                index = batch['index']-1
                filename = test_set.getImageID(index) + '.png'
                imgsize = Image.open(imgdir / filename).size
                if savepred: 
                    logging.info(f"Saving prediction of {filename}")
                    mask_pred_img = mask_to_image(mask_pred[0].cpu().numpy(), net.n_classes, imgsize)
                    mask_pred_img.save(outdir / filename)
                if visualize:
                    plot_img_and_mask_and_gt(image[0].cpu().numpy().transpose((1,2,0))[:,:,:3], mask_true[0].cpu().numpy(), mask_pred[0].cpu().numpy(), net.n_classes) 

    # Fixes a potential division by zero error
    if num_val_batches == 0:
        return dice_score
    return dice_score / num_val_batches


##########################################################################################

def get_args():
    parser = argparse.ArgumentParser(description='Predict masks from input images')
    parser.add_argument('--model', '-m', default=dir_checkpoint / ckp, metavar='FILE',
                        help='Specify the file in which the model is stored')
    parser.add_argument('--input', '-i', metavar='INPUT', default=imgdir, help='Directory of input images')
    parser.add_argument('--input_att', '-a', metavar='INPUT ATTENTION', default=attmapdir, help='Directory of input attention maps')
    parser.add_argument('--input_flow', '-of', metavar='INPUT OPTICAL FLOW', default=flowdir, help='Directory of input optical flows')
    parser.add_argument('--ground_truth', '-gt', metavar='GROUND TRUTH', default=gtdir, help='Directory of ground truth masks')
    parser.add_argument('--output', '-o', metavar='OUTPUT', default=outdir, help='Directory for output images')
    parser.add_argument('--viz', '-v', action='store_true', default=False, 
                        help='Visualize the images as they are processed')
    parser.add_argument('--mask_threshold', '-t', type=float, default=0.5,
                        help='Minimum probability value to consider a mask pixel white')
    parser.add_argument('--scale', '-s', type=float, default=1.0,
                        help='Scale factor for the input images')
    parser.add_argument('--bilinear', action='store_true', default=False, help='Use bilinear upsampling')
    parser.add_argument('--classes', '-c', type=int, default=2, help='Number of classes')
    parser.add_argument('--flow', action='store_true', default=False, help='Add optical flow to input')
    parser.add_argument('--pos', action='store_true', default=False, help='Add normalized position to input')
    parser.add_argument('--grayscale', '-gs', action='store_true', default=False, help='Convert RGB image to Greyscale for input')
    parser.add_argument('--noimg', action='store_true', default=False, help='No image as input')
    
    return parser.parse_args()

if __name__ == '__main__':
    args = get_args()
    logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
    # set_seed(0)

    useatt = True if args.input_att != None else False 
    useflow = True if (args.flow and args.input_flow != None) else False 

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
    net.load_state_dict(modelToLoad, strict=True)
    net.to(device=device)

    logging.info('Model loaded!')

    savepred = True if outdir != None else False 
    dice_score = test_net(
        net, 
        device=device,
        images_dir=imgdir, 
        masks_dir=gtdir, 
        attmaps_dir=attmapdir, 
        flows_dir=flowdir, 
        img_scale=args.scale,
        mask_threshold=args.mask_threshold, 
        useatt=useatt, 
        useflow=useflow, 
        addpositions=args.pos, 
        rgbtogs=args.grayscale, 
        noimg=args.noimg, 
        savepred=savepred, 
        visualize=args.viz)
        
    logging.info(f'Final average DICE score is: {dice_score}')