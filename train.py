import argparse
import logging
import sys
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
import wandb
from torch import optim
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm
import numpy as np 

from unet.unetutils.data_loading import BasicDataset, AttentionDataset
from unet.unetutils.dice_score import dice_loss
from evaluate import evaluate
from unet.unet_model import UNet, UNetAtt

# dir_img = Path('./data/imgs/')
dir_img = Path('D:\\Master Thesis\\data\\KU\\test')
# dir_img = Path('D:\\Master Thesis\\data\\GTEA\\GTEA\\train')
# dir_img = Path('D:\\Master Thesis\\data\\EgoHOS\\train\\image')

# dir_mask = Path('./data/masks/')
dir_mask = Path('D:\\Master Thesis\\data\\KU\\testannot')
# dir_mask = Path('D:\\Master Thesis\\data\\GTEA\\GTEA\\trainannot')
# dir_mask = Path('D:\\Master Thesis\\data\\EgoHOS\\train\\label')

dir_attmap = Path('./data/attmaps/')

dir_checkpoint = Path('./checkpoints')
# dir_checkpointwatt = Path('./checkpoints/attention/')
# dir_checkpointwoatt = Path('./checkpoints/woattention')


def train_net(net,
              device,
              epochs: int = 5,
              batch_size: int = 1,
              learning_rate: float = 1e-5,
              val_percent: float = 0.1,
              save_checkpoint: bool = False,
              save_best_checkpoint: bool = True,
              img_scale: float = 0.5,
              amp: bool = False, 
              useatt: bool = False, 
              lossframesdecay: bool = False, 
              addpositions: bool = False):
    # 1. Create dataset
    if useatt: 
        dataset = AttentionDataset(images_dir=dir_img, masks_dir=dir_mask, scale=img_scale, attmaps_dir=dir_attmap)
    else: 
        dataset = BasicDataset(images_dir=dir_img, masks_dir=dir_mask, scale=img_scale)

    # 2. Split into train / validation partitions
    n_val = int(len(dataset) * val_percent)
    n_train = len(dataset) - n_val
    train_set, val_set = random_split(dataset, [n_train, n_val], generator=torch.Generator().manual_seed(0))

    # 3. Create data loaders
    loader_args = dict(num_workers=4, pin_memory=True)
    train_loader = DataLoader(train_set, shuffle=True, batch_size=batch_size, **loader_args)
    val_loader = DataLoader(val_set, shuffle=False, drop_last=True, batch_size=min(batch_size, n_val), **loader_args)

    # (Initialize logging)
    project_name = 'U-Net'
    if useatt: 
        project_name += '-w-attention'
    if addpositions: 
        project_name += '-w-positions'
    experiment = wandb.init(project=project_name, resume='allow', anonymous='must')
    experiment.config.update(dict(epochs=epochs, batch_size=batch_size, learning_rate=learning_rate,
                                  val_percent=val_percent, save_checkpoint=save_checkpoint, img_scale=img_scale,
                                  amp=amp, use_attention=useatt))

    logging.info(f'''Starting training:
        Attention model: {useatt}
        Epochs:          {epochs}
        Batch size:      {batch_size}
        Learning rate:   {learning_rate}
        Training size:   {n_train}
        Validation size: {n_val}
        Checkpoints:     {save_checkpoint}
        Device:          {device.type}
        Images scaling:  {img_scale}
        Mixed Precision: {amp}
    ''')

    # 4. Set up the optimizer, the loss, the learning rate scheduler and the loss scaling for AMP
    # optimizer = optim.RMSprop(net.parameters(), lr=learning_rate, weight_decay=1e-2, momentum=0.9) # VANILLA SETTINGS: net.parameters(), lr=learning_rate, weight_decay=1e-8, momentum=0.9
    optimizer = optim.SGD(net.parameters(), lr=learning_rate, momentum=0.9)
    # scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'max', patience=2)  # goal: maximize Dice score
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda curr_epoch: (epochs - curr_epoch) / epochs) 
    grad_scaler = torch.cuda.amp.GradScaler(enabled=amp)
    criterion = nn.CrossEntropyLoss() # VANILLA 
    debug_criterion = nn.CrossEntropyLoss(reduction='none') ############## DEBUG CROSS ENTROPY
    # criterion = nn.BCELoss() # GROUND TRUTH = SOFT MAPS 
    global_step = 0

    # 5. Begin training
    for epoch in range(1, epochs+1):
        net.train()
        epoch_loss = 0
        epoch_dice = 0 
        with tqdm(total=n_train, desc=f'Epoch {epoch}/{epochs}', unit='img') as pbar:
            for batch in train_loader:
                images = batch['image']
                if addpositions: 
                    # Add normalized positions to input 
                    _, batchsize, w, h = images.shape
                    x = torch.tensor(np.arange(h)/(h-1))
                    y = torch.tensor(np.arange(w)/(w-1))
                    grid_x, grid_y = torch.meshgrid(x, y, indexing='ij')
                    grid_x = grid_x.repeat(len(images), 1, 1, 1)
                    grid_y = grid_y.repeat(len(images), 1, 1, 1)
                    images = torch.cat((images, grid_x, grid_y), dim=1)
                true_masks = batch['mask']
                index = batch['index']
                if useatt: 
                    attention_maps = batch['attmap']

                assert images.shape[1] == net.n_channels, \
                    f'Network has been defined with {net.n_channels} input channels, ' \
                    f'but loaded images have {images.shape[1]} channels. Please check that ' \
                    'the images are loaded correctly.'

                images = images.to(device=device, dtype=torch.float32)
                true_masks = true_masks.to(device=device, dtype=torch.long)
                index = index.to(device=device, dtype=torch.int)
                if useatt: 
                    attention_maps = attention_maps.to(device=device, dtype=torch.float32)

                with torch.cuda.amp.autocast(enabled=amp):
                    masks_pred = net(images)
                    if net.n_classes == 1:
                        loss = criterion(masks_pred.squeeze(1), true_masks.float())
                        loss += dice_loss(F.sigmoid(masks_pred.squeeze(1)), true_masks.float(), multiclass=False)
                        with torch.no_grad(): 
                            debug_loss = debug_criterion(masks_pred.squeeze(1), true_masks.float()) ################################# DEBUG CROSS ENTROPY 
                    else:
                        loss = criterion(masks_pred, true_masks)
                        loss += dice_loss(
                            F.softmax(masks_pred, dim=1).float(),
                            F.one_hot(true_masks, net.n_classes).permute(0, 3, 1, 2).float(),
                            multiclass=True
                        )
                        with torch.no_grad(): 
                            debug_loss = debug_criterion(masks_pred, true_masks) ################################# DEBUG CROSS ENTROPY 
                    
                    if lossframesdecay: 
                        loss /= index 

                optimizer.zero_grad(set_to_none=True)
                grad_scaler.scale(loss).backward()
                grad_scaler.step(optimizer)
                grad_scaler.update()

                pbar.update(images.shape[0])
                global_step += 1
                epoch_loss += loss.item()
                experiment.log({
                    'train loss': loss.item(),
                    'step': global_step,
                    'epoch': epoch
                })
                pbar.set_postfix(**{'loss (batch)': loss.item()})

            # Evaluation round (validation testing at the end of epoch)
            histograms = {}
            for tag, value in net.named_parameters():
                tag = tag.replace('/', '.')
                if not torch.isinf(value).any():
                    histograms['Weights/' + tag] = wandb.Histogram(value.data.cpu())
                if not torch.isinf(value.grad).any():
                    histograms['Gradients/' + tag] = wandb.Histogram(value.grad.data.cpu())

            val_score = evaluate(net, val_loader, device, useatt=useatt, addpos=addpositions)
            epoch_dice += val_score
            # scheduler.step(val_score)

            logging.info('Validation Dice score: {}'.format(val_score))
            experiment.log({
                'learning rate': optimizer.param_groups[0]['lr'],
                'validation Dice': val_score,
                'images': wandb.Image(images[0,:3].cpu()),
                'masks': {
                    'true': wandb.Image(true_masks[0].float().cpu()),
                    'pred': wandb.Image(masks_pred.argmax(dim=1)[0].float().cpu()),
                },
                'cross entropy': wandb.Image(debug_loss[0].float().cpu()), 
                'step': global_step,
                'epoch': epoch,
                **histograms
            })
        epoch_loss /= len(train_loader)
        scheduler.step() # Change learning rate 

        # 6. (Optional) Save checkpoint at each epoch 
        
        if save_checkpoint or save_best_checkpoint:
            adddirckp = 'U-Net-' + str(net.n_channels)
            if useatt: 
                adddirckp += '-w-attention' 
            if addpositions: 
                adddirckp += '-w-positions'
            dirckp = dir_checkpoint / adddirckp
            dirckp.mkdir(parents=True, exist_ok=True)


        if save_checkpoint:            
            torch.save(net.state_dict(), str(dirckp / 'checkpoint_epoch{}.pth'.format(epoch)))
            logging.info(f'Checkpoint {epoch} saved!')
        
        # 7. (Optional) Save best model 
        if save_best_checkpoint: 
            if epoch == 1: 
                best_loss = epoch_loss 
                best_ckpt = 1 
                if not save_checkpoint: 
                    torch.save(net.state_dict(), dirckp / 'checkpoint_epoch_best.pth')
                    logging.info(f'Checkpoint {epoch} saved!')
            else: 
                if epoch_loss < best_loss: 
                    best_loss = epoch_loss
                    best_ckpt = epoch
                    torch.save(net.state_dict(), dir_checkpoint / 'checkpoint_epoch_best.pth')
                    logging.info(f'Best checkpoint at {epoch} saved!')
            
            logging.info('Epoch loss: {}'.format(best_loss))

        # 8. Log all the previous parameters 
        experiment.log({
            'epoch': epoch, 
            'best epoch': best_ckpt, 
            'train loss epoch avg': epoch_loss, 
        })
    
    return best_ckpt 


def get_args():
    parser = argparse.ArgumentParser(description='Train the UNet on images and target masks')
    parser.add_argument('--epochs', '-e', metavar='E', type=int, default=5, help='Number of epochs')
    parser.add_argument('--batch-size', '-b', dest='batch_size', metavar='B', type=int, default=1, help='Batch size')
    parser.add_argument('--learning-rate', '-l', metavar='LR', type=float, default=1e-5,
                        help='Learning rate', dest='lr')
    parser.add_argument('--load', '-f', type=str, default=False, help='Load model from a .pth file')
    parser.add_argument('--scale', '-s', type=float, default=1.0, help='Downscaling factor of the images')
    parser.add_argument('--validation', '-v', dest='val', type=float, default=10.0,
                        help='Percent of the data that is used as validation (0-100)')
    parser.add_argument('--amp', action='store_true', default=False, help='Use mixed precision')
    parser.add_argument('--bilinear', action='store_true', default=False, help='Use bilinear upsampling')
    parser.add_argument('--classes', '-c', type=int, default=2, help='Number of classes')
    parser.add_argument('--attention', action='store_true', default=False, help='Use UNet with attention model')
    parser.add_argument('--framesdecay', action='store_true', default=False, help='Modify loss function to add the frames lack of importance')
    parser.add_argument('--saveall', action='store_true', default=False, help='Save checkpoint at each epoch')
    parser.add_argument('--savebest', action='store_false', default=True, help='Save checkpoint of best epoch')
    parser.add_argument('--wpos', action='store_true', default=False, help='Add normalized position to input')

    return parser.parse_args()


if __name__ == '__main__':
    args = get_args()

    logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logging.info(f'Using device {device}')

    # Change here to adapt to your data
    # n_channels=3 for RGB images
    # n_classes is the number of probabilities you want to get per pixel
    # if the model with attention is used, a different model will be loaded 
    n_channels = 3 
    if args.wpos: 
        n_channels = 5 
    if args.attention: 
        net = UNetAtt(n_channels=n_channels, n_classes=args.classes, bilinear=args.bilinear)
    else: 
        net = UNet(n_channels=n_channels, n_classes=args.classes, bilinear=args.bilinear)

    logging.info(f'Network:\n'
                 f'\t{net.n_channels} input channels\n'
                 f'\t{net.n_classes} output channels (classes)\n'
                 f'\t{"Bilinear" if net.bilinear else "Transposed conv"} upscaling')

    if args.load:
        modelToLoad = torch.load(args.load, map_location=device)
        net.load_state_dict(modelToLoad, strict=False)
        logging.info(f'Model loaded from {args.load}')

    net.to(device=device)
    try:
        best_ckpt = train_net(
            net=net,
            epochs=args.epochs,
            batch_size=args.batch_size,
            learning_rate=args.lr,
            device=device,
            img_scale=args.scale,
            val_percent=args.val / 100,
            save_checkpoint=args.saveall,
            save_best_checkpoint=args.savebest,
            amp=args.amp, 
            useatt=args.attention, 
            lossframesdecay=args.framesdecay, 
            addpositions=args.wpos)
        logging.info(f'Best model is found at checkpoint #{best_ckpt}.')
    except KeyboardInterrupt:
        torch.save(net.state_dict(), 'INTERRUPTED.pth')
        logging.info('Saved interrupt')
        raise