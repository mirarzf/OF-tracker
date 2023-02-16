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

import matplotlib.pyplot as plt 

from torch import Tensor 
from unet.unetutils.dice_score import multiclass_dice_coeff, dice_coeff

import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'

# CHOOSE INPUT DIRECTORIES 
imgdir = Path('D:\\Master Thesis\\data\\KU\\train')
imgdir = Path('D:\\Master Thesis\\data\\KU\\test')
# imgdir = Path("./data/imgs")
imgfilenames = [f for f in imgdir.glob('*.png') if f.is_file()] 
gtdir = Path('D:\\Master Thesis\\data\\KU\\trainannot')
gtdir = Path('D:\\Master Thesis\\data\\KU\\testannot')
# gtdir = Path("./data/masks")
attmapdir = None # Path("./")
# attmapdir = Path('D:\\Master Thesis\\data\\KU\\testannot')
# attmapdir = Path("./data/attmaps")
outdir = Path("./results/unet")

dir_checkpoint = Path('./checkpoints')
# ckp = "U-Net-3/tKU_bs16_e50_lr1e-2.pth" ## GIVING THE CORRECT OUTPUT 
# ckp = "U-Net-3/tKU_bs16_e50_lr1e-2_1.pth" # GIVING THE WRONG OUTPUT 
ckp = "U-Net-3/checkpoint_epoch_best.pth"

def normalizeToRGB(array): 
    mini, maxi = np.min(array), np.max(array)
    array = (array-mini)/(maxi-mini)*255 
    array = array.astype(int)
    return array

def normalizeToRGBtensor(tens): 
    for i in range(tens.shape[1]): 
        array = tens[:,i]
        mini, maxi = torch.min(array), torch.max(array)
        array = (array-mini)/(maxi-mini)*255 
        tens[:,i] = array
    return tens

def predict_img(net,
                full_img,
                device,
                scale_factor=1,
                out_threshold=0.5, 
                useatt: bool = False, 
                full_attmap = None, 
                addpositions: bool = False):
    img = torch.from_numpy(AttentionDataset.preprocess(full_img, scale_factor, is_mask=False))
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

    assert img.shape[1] == net.n_channels, \
        f'Network has been defined with {net.n_channels} input channels, ' \
        f'but loaded images have {img.shape[1]} channels. Please check that ' \
        'the images are loaded correctly.'

    if useatt: 
        attmap = torch.from_numpy(AttentionDataset.preprocess(full_attmap, scale_factor, is_mask=True))
        attmap = attmap.unsqueeze(0)
        attmap = attmap.to(device=device, dtype=torch.float32)

    with torch.no_grad():
        if useatt: 
            output = net(img, attmap)
        else: 
            output = net(img)
        # PRINT HISTOGRAMS OF THE VALUES -- DEBUG HISTOGRAMS ########################################################## 
        class0 = output[0,0].cpu().numpy()
        class1 = output[0,1].cpu().numpy()
        plt.hist(class0.flatten())
        plt.hist(class1.flatten())
        plt.title("output")
        plt.show()
        plt.imshow(normalizeToRGB(class0))
        plt.colorbar()
        plt.title("output class 0")
        plt.show()
        plt.imshow(normalizeToRGB(class1))
        plt.colorbar()
        plt.title("output class 1")
        plt.show()
        output = normalizeToRGBtensor(output)

        if net.n_classes > 1:
            # print(output)
            probs = F.softmax(output, dim=1).float()
            # probs = output
            print("probs size", probs.size())
            # print(probs[:,150:155,150:155])
        else:
            probs = torch.sigmoid(output)
            # print(probs[:,150:155,150:155])
        plt.hist(probs[0,0].cpu().numpy().flatten())
        plt.hist(probs[0,1].cpu().numpy().flatten())
        plt.title("probs")
        plt.show()
        print(torch.max(probs), torch.min(probs))

        # tf = transforms.Compose([
        #     transforms.ToPILImage(),
        #     transforms.Resize((full_img.size[1], full_img.size[0])),
        #     transforms.ToTensor()
        # ])

        # full_mask = tf(probs.cpu()).squeeze()
        full_mask = F.interpolate(probs, (full_img.size[1], full_img.size[0]), mode='nearest-exact').cpu().squeeze()
        plt.hist(full_mask[0].numpy().flatten())
        plt.hist(full_mask[1].numpy().flatten())
        plt.title("full_mask")
        print(full_mask.size())
        plt.show()
        print(torch.max(probs), torch.min(probs))

    if net.n_classes == 1:
        return (full_mask > out_threshold).float().numpy()
    else:
        return full_mask.argmax(dim=0).float().numpy()


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

def mask_to_image(mask: np.ndarray):
    if mask.ndim == 2:
        return Image.fromarray((mask * 255).astype(np.uint8))
    elif mask.ndim == 3:
        return Image.fromarray((np.argmax(mask, axis=0) * 255 / mask.shape[0]).astype(np.uint8))

def diceuniqueclass(pred, gt, classvalue = 1): 
    assert pred.shape[0] == gt.shape[0] and pred.shape[1] == gt.shape[1], \
        f'Shape of prediction is {pred.shape} and shape of prediction is {gt.shape}'
    gtones = np.where(gt == classvalue, 1, 0)
    cardpred = np.sum(pred)
    cardgt = np.sum(gtones)
    inter = np.sum(pred*gtones)
    return 2 * inter / (cardpred + cardgt)

# def dice(pred, gt, multiclass = True): 
#     ''' 
#     Input is in shape [C, H, W] with C equals to number of classes 
#     '''
#     meandice = 0 
#     if multiclass: 
#         for i in range(1,pred.shape[0]): 
#             meandice += diceuniqueclass(pred[i], gt, i)
#         meandice /= pred.shape[0]
#     else: 
#         meandice = diceuniqueclass(pred, gt, 1)
#     return meandice 

def dice(input: Tensor, target: Tensor, multiclass: bool = True):
    # Dice loss (objective to minimize) between 0 and 1
    assert input.size() == target.size()
    fn = multiclass_dice_coeff if multiclass else dice_coeff
    return fn(input, target, reduce_batch_first=True)


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
        if useatt: 
            attmap = Image.open(in_files_att[i])
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
        plt.imshow(mask)
        plt.colorbar()
        plt.title("mask")
        plt.show()
        
        gt = Image.open(in_files_gt[i])
        gt = np.asarray(gt)
        thres = 0
        print(gt.shape)
        gt = np.where(gt > thres, 1, 0)[:,:,0]
        print(gt.shape)
        print(gt[:5,:5])

        # dice_score += dice(
        #     F.one_hot(mask, net.n_classes).permute(2, 0, 1), 
        #     gt, 
        #     multiclass=(net.n_classes > 1))

        if net.n_classes == 1: 
            dice_score += dice(torch.from_numpy(mask).float(), torch.from_numpy(gt)[None, ...])
        else: 
            onehotmask = F.one_hot(torch.from_numpy(mask)[None, ...].long(), net.n_classes).permute(0, 3, 1, 2).float()
            onehotgt = F.one_hot(torch.from_numpy(gt)[None, ...].long(), net.n_classes).permute(0, 3, 1, 2).float()
            dice_score += dice(onehotmask[:, 1:, ...], onehotgt[:, 1:, ...])
        

        if not args.no_save:
            out_filename = out_files[i]
            result = mask_to_image(mask)
            result.save(out_filename)
            logging.info(f'Mask saved to {out_filename}')

        if args.viz:
            logging.info(f'Visualizing results for image {filename}, close to continue...')
            plot_img_and_mask_and_gt(img, gt, mask)
    
    dice_score /= len(in_files)

    logging.info(f'Final average DICE score is: {dice_score}')