import torch
import torch.nn.functional as F
from tqdm import tqdm

import numpy as np 

from unet.unetutils.dice_score import multiclass_dice_coeff, dice_coeff

# REPRODUCIBILITY 
import random
def set_seed(seed: int = 42) -> None:
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    # When running on the CuDNN backend, two further options must be set
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    print(f"Random seed set as {seed}")
set_seed(0)
# END REPRODUCIBILLITY 

def evaluate(net, dataloader, device, useatt=False, addpos=False):
    net.eval()
    num_val_batches = len(dataloader)
    dice_score = 0

    # iterate over the validation set
    for batch in tqdm(dataloader, total=num_val_batches, desc='Validation round', unit='batch', leave=False):
        image, mask_true = batch['image'], batch['mask']
        if addpos: 
            # Add absolute positions to input 
            batchsize, _, w, h = image.shape
            x = torch.tensor(np.arange(h)/(h-1))
            y = torch.tensor(np.arange(w)/(w-1))
            grid_x, grid_y = torch.meshgrid(x, y, indexing='ij')
            grid_x = grid_x.repeat(len(image), 1, 1, 1)
            grid_y = grid_y.repeat(len(image), 1, 1, 1)
            image = torch.cat((image, grid_x, grid_y), dim=1)
        if useatt: 
            attmap = batch['attmap']
            attmap = attmap.to(device=device, dtype=torch.float32)
        # move images and labels to correct device and type
        image = image.to(device=device, dtype=torch.float32)
        mask_true = mask_true.to(device=device, dtype=torch.long)
        mask_true = F.one_hot(mask_true, net.n_classes).permute(0, 3, 1, 2).float()

        with torch.no_grad():
            # predict the mask
            if useatt: 
                mask_pred = net(image, attmap)
            else: 
                mask_pred = net(image)

            # convert to one-hot format
            if net.n_classes == 1:
                mask_pred = (F.sigmoid(mask_pred) > 0.5).float()
                # compute the Dice score
                dice_score += dice_coeff(mask_pred, mask_true, reduce_batch_first=False)
            else:
                mask_pred = F.one_hot(mask_pred.argmax(dim=1), net.n_classes).permute(0, 3, 1, 2).float()
                # compute the Dice score, ignoring background
                dice_score += multiclass_dice_coeff(mask_pred[:, 1:, ...], mask_true[:, 1:, ...], reduce_batch_first=False)

        
    net.train()

    # Fixes a potential division by zero error
    if num_val_batches == 0:
        return dice_score
    return dice_score / num_val_batches
