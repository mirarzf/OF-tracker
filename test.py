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


import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'

# CHOOSE INPUT DIRECTORIES 
# imgdir = Path("../data/GTEA/frames")
imgdir = Path('D:\\Master Thesis\\data\\KU\\train')
# imgdir = Path('D:\\Master Thesis\\data\\KU\\test')
imgfilenames = [f for f in imgdir.glob('*.png') if f.is_file()] 
# imgdir = Path("./data/imgs")
gtdir = Path('D:\\Master Thesis\\data\\KU\\trainannot')
# gtdir = Path('D:\\Master Thesis\\data\\KU\\testannot')
attmapdir = None # Path("./")
# attmapdir = Path('D:\\Master Thesis\\data\\KU\\testannot')
# attmapdir = Path("./data/attmaps")
outdir = Path("./results/unet")

dir_checkpoint = Path('./checkpoints')
ckp = "U-Net-5-w-positions/tKU_bs16_e50_lr5e-1.pth" 
# ckp = "U-Net-3/tKU_bs16_e50_lr1e-1.pth" 


def predict_img(net,
                full_img,
                device,
                scale_factor=1,
                out_threshold=0.5, 
                useatt: bool = False, 
                full_attmap = None, 
                addpositions: bool = False):
    net.eval()
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

    if useatt: 
        attmap = torch.from_numpy(AttentionDataset.preprocess(full_attmap, scale_factor, is_mask=True))
        attmap = attmap.unsqueeze(0)
        attmap = attmap.to(device=device, dtype=torch.float32)

    with torch.no_grad():
        if useatt: 
            output = net(img, attmap)
        else: 
            output = net(img)

        if net.n_classes > 1:
            probs = F.softmax(output, dim=1)[0]
        else:
            probs = torch.sigmoid(output)[0]

        tf = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((full_img.size[1], full_img.size[0])),
            transforms.ToTensor()
        ])

        full_mask = tf(probs.cpu()).squeeze()

    if net.n_classes == 1:
        return (full_mask > out_threshold).numpy()
    else:
        return F.one_hot(full_mask.argmax(dim=0), net.n_classes).permute(2, 0, 1).numpy()


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
    assert pred.shape[0] == gt.shape[0] and pred.shape[1] == gt.shape[1]
    predones = np.where(pred == classvalue, 1, 0)
    gtones = np.where(gt == classvalue, 1, 0)
    cardpred = np.sum(predones)
    cardgt = np.sum(gtones)
    inter = np.sum(predones*gtones)
    return 2 * inter / (cardpred + cardgt)

def dice(pred, gt, multiclass = True): 
    ''' 
    Input is in shape [C, H, W] with C equals to number of classes 
    '''
    meandice = 0 
    if multiclass: 
        for i in range(pred.shape[0]): 
            meandice += diceuniqueclass(pred[i], gt, i)
    else: 
        meandice = diceuniqueclass(pred, gt, 1)
    return meandice 


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
    net.load_state_dict(modelToLoad, strict=False)
    net.to(device=device)

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
        
        gt = Image.open(in_files_gt[i])
        gt = np.asarray(gt)
        thres = 0
        gt = np.where(gt > thres, 1, 0)[:,:,0]

        dice_score += dice(
            mask, 
            gt, 
            multiclass=net.n_classes > 1)

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