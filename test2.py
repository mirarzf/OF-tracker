import argparse
import logging
from pathlib import Path
from tqdm import tqdm

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms

from unet.unetutils.data_loading import AttentionDataset
from unet.unet_model import UNet, UNetAtt
from unet.unetutils.utils import plot_img_and_mask_and_gt

from unet.unetutils.dice_score import multiclass_dice_coeff, dice_coeff

import matplotlib.pyplot as plt 

import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'

# CHOOSE INPUT DIRECTORIES 
# imgdir = Path("../data/GTEA/frames")
# imgdir = Path('D:\\Master Thesis\\data\\KU\\train')
imgdir = Path('D:\\Master Thesis\\data\\KU\\test')
imgfilenames = [f for f in imgdir.glob('*.png') if f.is_file()] 
# imgdir = Path("./data/imgs")
# gtdir = Path('D:\\Master Thesis\\data\\KU\\trainannot')
gtdir = Path('D:\\Master Thesis\\data\\KU\\testannot')
attmapdir = None # Path("./")
# attmapdir = Path('D:\\Master Thesis\\data\\KU\\testannot')
# attmapdir = Path("./data/attmaps")
outdir = Path("./results/unet")

dir_checkpoint = Path('./checkpoints')
ckp = "U-Net-5-w-positions/checkpoint_epoch_best.pth" 
ckp = "U-Net-3/checkpoint_epoch_best.pth" 
# ckp = "U-Net-5-w-positions/tKU_bs16_e50_lr5e-1_1.pth" 
ckp = "U-Net-3/tKU_bs16_e50_lr1e-2.pth" 
ckp = "U-Net-3/checkpoint_botched.pth" # bs16_e50_lr1e-2_2


def predict_img(net,
                full_img,
                device,
                scale_factor=1,
                out_threshold=0.5, 
                useatt: bool = False, 
                full_attmap = None, 
                addpositions: bool = False):
    img = torch.from_numpy(full_img)
    if addpositions: 
        # Add normalized positions to input 
        _, w, h = img.shape
        x = torch.tensor(np.arange(h)/(h-1))
        y = torch.tensor(np.arange(w)/(w-1))
        grid_x, grid_y = torch.meshgrid(x, y, indexing='ij')
        grid_x = grid_x.repeat(1, 1, 1)
        grid_y = grid_y.repeat(1, 1, 1)
        img = torch.cat((img, grid_x, grid_y), dim=0)
    img = img.unsqueeze(0)
    img = img.to(device=device, dtype=torch.float32)

    if useatt: 
        attmap = torch.from_numpy(attmap)
        attmap = attmap.unsqueeze(0)
        attmap = attmap.to(device=device, dtype=torch.float32)

    with torch.no_grad():
        if useatt: 
            output = net(img, attmap)
        else: 
            output = net(img)
        # PRINT HISTOGRAMS OF THE VALUES -- DEBUG HISTOGRAMS ########################################################## 
        plt.hist(output[:,0].cpu().numpy().flatten(), bins=256)
        plt.hist(output[:,1].cpu().numpy().flatten(), bins=256)
        plt.show()
        print(output.size(), "c'est la size du output")
        print(f"min: {torch.min(output)} et max: {torch.max(output)}")

        # convert to one-hot format
        if net.n_classes == 1:
            mask_pred = (F.sigmoid(output) > out_threshold).float()
        else:
            mask_pred = F.softmax(output, dim=1).float()
            # mask_pred = output 
            # for i in range(net.n_classes): 
            #     mask_pred[:,i, ...] = F.sigmoid(output[:,i, ...])
            mask_pred = F.one_hot(mask_pred.argmax(dim=1), net.n_classes).permute(0, 3, 1, 2).float()
        
        # PRINT HISTOGRAMS OF THE VALUES -- DEBUG HISTOGRAMS ########################################################## 
        plt.hist(mask_pred[:,0].cpu().numpy().flatten(), bins=256)
        plt.hist(mask_pred[:,1].cpu().numpy().flatten(), bins=256)
        plt.show()
        return mask_pred.cpu()



def get_args():
    parser = argparse.ArgumentParser(description='Predict masks from input images')
    parser.add_argument('--model', '-m', default=dir_checkpoint / ckp, metavar='FILE',
                        help='Specify the file in which the model is stored')
    inputgroup = parser.add_mutually_exclusive_group(required=True)
    inputgroup.add_argument('--input', '-i', metavar='INPUT', nargs='+', help='Filenames of input images')
    inputgroup.add_argument('--ground_truth', '-gt', metavar='GROUND TRUTH', nargs='+', help='Filenames of ground truth masks')
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
    parser.add_argument('--input_att', '-a', metavar='INPUT ATTENTION', nargs='+', help='Filenames of input attention maps')
    parser.add_argument('--wpos', action='store_true', default=False, help='Add normalized position to input')

    return parser.parse_args()


def get_output_filenames(args): 
    return args.output or [outdir / f.name for f in imgfilenames]
    
def get_attmap_filenames(args): 
    if args.dir and attmapdir != None: 
        return [attmapdir / f.name for f in imgfilenames]
    else: 
        return args.input_att 

def mask_to_image(mask: np.ndarray, size: tuple):
    print(mask.shape, "ihi")
    if mask.ndim == 2:
        return Image.fromarray((mask * 255).astype(np.uint8)).resize(size)
    elif mask.ndim == 3:
        return Image.fromarray((np.argmax(mask, axis=0) * 255 / mask.shape[0]).astype(np.uint8)).resize(size)

if __name__ == '__main__':
    args = get_args()
    logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')

    # Prepare the input files 
    if args.dir: 
        in_files = imgfilenames
        in_files_gt = [gtdir / f.name for f in imgfilenames]
    else: 
        in_files = args.input
        in_files_gt = args.ground_truth
    in_files_att = get_attmap_filenames(args)
    out_files = get_output_filenames(args)

    useatt = True if in_files_att != None else False 

    n_channels = 3 
    if args.wpos: 
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
    net.eval()

    logging.info('Model loaded!')

    dice_score = 0

    for i, filename in enumerate(tqdm(in_files)):
        logging.info(f'\nPredicting image {filename} ...')
        img = Image.open(filename)
        originalsize = img.size
        print(img.size)
        img = AttentionDataset.preprocess(img, scale=args.scale, is_mask=False)
        if useatt: 
            attmap = Image.open(in_files_att[i])
            attmap = AttentionDataset.preprocess(attmap, scale=args.scale, is_mask=True)
        else: 
            attmap = None 

        mask = predict_img(net=net,
                           full_img=img,
                           scale_factor=args.scale,
                           out_threshold=args.mask_threshold,
                           device=device, 
                           useatt=useatt, 
                           full_attmap=attmap, 
                           addpositions=args.wpos)
        
        gt = Image.open(in_files_gt[i])
        gt = AttentionDataset.preprocess(gt, scale=args.scale, is_mask=True)
        gt = gt[np.newaxis,:,:]
        print("that gt shape", gt.shape)
        gtdice = torch.as_tensor(gt.copy()).long().contiguous()
        gtdice = F.one_hot(gtdice, net.n_classes).permute(0, 3, 1, 2).float()

        print("yo mask size BIS", mask.shape)

        if net.n_classes == 1: 
            dice_score += dice_coeff(mask, gtdice, reduce_batch_first=False)
        else: 
            dice_score += multiclass_dice_coeff(mask[:,1:, ...], gtdice[:,1:, ...], reduce_batch_first=False)
        
        mask = np.argmax(mask.squeeze().numpy(), axis=0)

        if not args.no_save:
            out_filename = out_files[i]
            result = mask_to_image(mask, originalsize)
            result.save(out_filename)
            logging.info(f'Mask saved to {out_filename}')

        if args.viz:
            logging.info(f'Visualizing results for image {filename}, close to continue...')
            plot_img_and_mask_and_gt(img, gt, mask)
    
    dice_score /= len(in_files)

    logging.info(f'Final average DICE score is: {dice_score}')