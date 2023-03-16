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
from unet.unetutils.utils import plot_img_and_mask
from test import mask_to_image

import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'


# CHOOSE INPUT DIRECTORIES 
# imgdir = Path("../data/GTEA/frames")
# imgdir = Path('D:\\Master Thesis\\data\\KU\\train')
# imgdir = Path('D:\\Master Thesis\\data\\KU\\test')
imgdir = Path("./data/test/imgs")
imgfilenames = [f for f in imgdir.iterdir() if f.is_file() and 'mirrored' not in f.name] 
attmapdir = None # Path("./")
# attmapdir = Path('D:\\Master Thesis\\data\\KU\\testannot')
# attmapdir = Path("./data/attmaps")
outdir = Path("./results/unet")

dir_checkpoint = Path('./checkpoints')
ckp = "U-Net-5-w-positions/tKU_bs16_e50_lr1e-1_old.pth" 


def predict_img(net,
                full_img: Image,
                out_filename: str, 
                device,
                img_scale: float = 1,
                mask_threshold: float =0.5, 
                useatt: bool = False, 
                full_attmap: Image = None, 
                addpositions: bool = False, 
                savepred: bool = True, 
                visualize: bool = False):
    net.eval()
    imgsize = full_img.size
    img = torch.from_numpy(AttentionDataset.preprocess(full_img, img_scale, is_mask=False))
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
        attmap = torch.from_numpy(AttentionDataset.preprocess(full_attmap, img_scale, is_mask=True))
        attmap = attmap.unsqueeze(0)
        attmap = attmap.to(device=device, dtype=torch.float32)

    with torch.no_grad():
        # predict the mask 
        if useatt: 
            mask_pred = net(img, attmap)
        else: 
            mask_pred = net(img)
### 
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
            plot_img_and_mask(img[0].cpu().numpy().transpose((1,2,0)), mask_pred[0].cpu().numpy(), net.n_classes) 
            logging.info(f'Visualizing results for image {filename}, close to continue...')
            

def get_args():
    parser = argparse.ArgumentParser(description='Predict masks from input images')
    parser.add_argument('--model', '-m', default=dir_checkpoint / ckp, metavar='FILE',
                        help='Specify the file in which the model is stored')
    inputgroup = parser.add_mutually_exclusive_group(required=True)
    inputgroup.add_argument('--input', '-i', metavar='INPUT', nargs='+', help='Filenames of input images')
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

# def mask_to_image(mask: np.ndarray):
#     if mask.ndim == 2:
#         return Image.fromarray((mask * 255).astype(np.uint8))
#     elif mask.ndim == 3:
#         return Image.fromarray((np.argmax(mask, axis=0) * 255 / mask.shape[0]).astype(np.uint8))


if __name__ == '__main__':
    args = get_args()
    logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')

    # Prepare the input files 
    if args.dir: 
        in_files = imgfilenames
    else: 
        in_files = args.input
    in_files_att = get_attmap_filenames(args)
    out_files = get_output_filenames(args)

    useatt = True if in_files_att != None else False 

    n_channels = 3 
    if args.wpos: 
        n_channels = 5 
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
        
        out_filename = out_files[i]
        predict_img(net=net,
                    full_img=img,
                    out_filename=out_filename, 
                    img_scale=args.scale,
                    mask_threshold=args.mask_threshold,
                    device=device, 
                    useatt=useatt, 
                    full_attmap=attmap, 
                    addpositions=args.wpos, 
                    savepred=not args.no_save, 
                    visualize=args.viz)

        # if not args.no_save:
        #     out_filename = out_files[i]
        #     result = mask_to_image(mask)
        #     result.save(out_filename)
        #     logging.info(f'Mask saved to {out_filename}')

        # if args.viz:
        #     logging.info(f'Visualizing results for image {filename}, close to continue...')
        #     plot_img_and_mask(img, mask)