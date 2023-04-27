import argparse
import logging
from pathlib import Path

import numpy as np 
import torch
import torch.nn as nn
import torch.nn.functional as F
import wandb
from torch import optim
from torch.utils.data import DataLoader, random_split
import albumentations as A
from tqdm import tqdm

from unet.unetutils.data_loading import MasterDataset
from unet.unetutils.dice_score import dice_loss
from evaluate import evaluate
from test import test_net
from unet.unet_model import UNet, UNetAtt

from copy import deepcopy

from test import test_net

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
    logging.info(f"Random seed set as {seed}. \n")
# END REPRODUCIBILLITY 

# DATA DIRECTORIES 
## FOR TRAINING 
dir_img = Path('./data/imgs/')
# dir_img = Path('D:\\Master Thesis\\data\\KU\\train')
# dir_img = Path('D:\\Master Thesis\\data\\GTEA\\GTEA\\train')
# dir_img = Path('D:\\Master Thesis\\data\\EgoHOS\\train\\image')

dir_mask = Path('./data/masks/')
# dir_mask = Path('D:\\Master Thesis\\data\\KU\\trainannot')
# dir_mask = Path('D:\\Master Thesis\\data\\GTEA\\GTEA\\trainannot')
# dir_mask = Path('D:\\Master Thesis\\data\\EgoHOS\\train\\label')

dir_attmap = Path('./data/attmaps/')

dir_flo = Path('./data/flows/')

## FOR TESTING 
dir_img_test = Path('./data/test/imgs/')
dir_mask_test = Path('./data/test/masks')
dir_attmap_test = Path('./data/test/attmaps')
dir_flow_test = Path('./data/test/flows')

## PARENT FOLDER OF CHECKPOINTS 
dir_checkpoint = Path('./checkpoints')

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
              useflow: bool = False,
              lossframesdecay: bool = False, 
              addpositions: bool = False, 
              rgbtogs: bool = False, 
              foldnumber: int = 0):
    
    # 1. Choose data augmentation transforms (using albumentations) 
    geotransform = A.Compose([ 
        A.HorizontalFlip(p=0.5)
    ], 
    additional_targets={'attmap':'mask'})
    colortransform = A.Compose([ 
        A.RandomBrightnessContrast(p=0.5)
    ])
    dataaugtransform = {'geometric': geotransform, 
                        'color': colortransform}
    # dataaugtransform = dict() ################################################### COMMENT IF YOU WANT DATA AUGMENTATION 

    # 2. Split into train / validation partitions
    ids = [file.stem for file in dir_img.iterdir() if file.is_file() and file.name != '.gitkeep']
    n_ids = len(ids)
    data_indices = list(range(n_ids))
    np.random.shuffle(data_indices)
    # Create folds if validation percentage is not 0 
    n_val = int(n_ids * val_percent)
    if n_val != 0: 
        k_number = n_ids // n_val
        last_idx_of_split = []
        q = n_ids // k_number 
        r = n_ids % k_number
        for i in range(k_number): 
            if i < r: 
                last_idx_of_split.append(i*q+1)
            else: 
                last_idx_of_split.append((i+1)*q)
        last_idx_of_split.append(n_ids)
        # Current fold number is (between [0;k-1]): 
        train_ids = [ids[idx] for idx in data_indices[:last_idx_of_split[foldnumber]]+data_indices[last_idx_of_split[foldnumber+1]:]] 
        val_ids = [ids[idx] for idx in data_indices[last_idx_of_split[foldnumber]:last_idx_of_split[foldnumber+1]]] 
    else: 
        train_ids = ids 
        val_ids = [] 
    # ### SELECT IDs FOR SEQUENCE TRAINING ### 
    # banned_id = "green0410_0452"
    # train_ids = [id for id in ids if banned_id not in id]
    # val_ids = [id for id in ids if banned_id in id]
    # train_ids = [ids[i] for i in data_indices if banned_id not in ids[i]]
    # # val_ids = [id for id in ids if banned_id not in id]
    # # train_ids = [id for id in ids if banned_id in id]
    # # train_ids = [ids[i] for i in data_indices if banned_id in ids[i]]
    # ### END OF SELECT IDs FOR SEQUENCE TRAINING ###
    ### SELECT IDs FOR HAND PICKED VALIDATION SET ### 
    val_ids = ["0838_0917_extract_10", 
             "0838_0917_extract_100", 
             "0838_0917_extract_400", 
             "2108_2112_extract_110",
             "5909_5915_extract_10", 
             "5909_5915_extract_70", 
             "5909_5915_extract_140", 
             "green0410_0452_extract_750", 
             "green0410_0452_extract_800", 
             "green0410_0452_extract_1000"]
    train_ids = [id for id in ids if id not in val_ids]
    ### END OF SELECT IDs FOR HAND PICKED VALIDATION SET ### 
    n_train = len(train_ids)
    n_val = len(val_ids)
    val_percent = round(n_val/n_train,2) 
    logging.info(f'''Validation dataset contains following ids: {val_ids}''')

    # 3. Create datasets
    train_set = MasterDataset(images_dir=dir_img, masks_dir=dir_mask, file_ids=train_ids, scale=img_scale, transform=dataaugtransform, attmaps_dir=dir_attmap, withatt=useatt, flo_dir=dir_flo, withflo=useflow, grayscale=rgbtogs) 
    val_set = MasterDataset(images_dir=dir_img, masks_dir=dir_mask, file_ids=val_ids, scale=img_scale, transform=dataaugtransform, attmaps_dir=dir_attmap, withatt=useatt, flo_dir=dir_flo, withflo=useflow, grayscale=rgbtogs) 

    print(len(val_set)) ############################ DEBUG PRINT 

    # 4. Create data loaders
    loader_args = dict(num_workers=4, pin_memory=True)
    train_loader = DataLoader(train_set, shuffle=True, batch_size=batch_size, **loader_args)
    val_loader = DataLoader(val_set, shuffle=False, batch_size=batch_size, **loader_args)

    # (Initialize logging)
    project_name = "OF-Tracker-TBDeleted"
    experiment = wandb.init(project=project_name, resume='allow', anonymous='must')
    experiment.config.update(dict(
        epochs=epochs, 
        batch_size=batch_size, 
        learning_rate=learning_rate, 
        val_percent=val_percent, 
        save_checkpoint=save_checkpoint, 
        img_scale=img_scale, 
        amp=amp, 
        use_attention=useatt, 
        use_opticalflow=useflow, 
        use_positions=addpositions, 
        augmented_data=(True if 'geometric' in dataaugtransform.keys() else False), 
        validation_size=n_val, 
        training_size = n_train
        ))

    logging.info(f'''Starting training:
        Attention model: {useatt}
        Optical Flow input: {useflow}
        Positions input: {addpositions}
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

    # 5. Set up the optimizer, the loss, the learning rate scheduler and the loss scaling for AMP
    # optimizer = optim.RMSprop(net.parameters(), lr=learning_rate, weight_decay=1e-2, momentum=0.9) # VANILLA SETTINGS: net.parameters(), lr=learning_rate, weight_decay=1e-8, momentum=0.9
    optimizer = optim.SGD(net.parameters(), lr=learning_rate, momentum=0.9)
    # scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'max', patience=2)  # goal: maximize Dice score
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda curr_epoch: (epochs - curr_epoch) / epochs) 
    grad_scaler = torch.cuda.amp.GradScaler(enabled=amp)
    criterion = nn.CrossEntropyLoss() # VANILLA 
    debug_criterion = nn.CrossEntropyLoss(reduction='none') ############## DEBUG CROSS ENTROPY
    # criterion = nn.BCELoss() # GROUND TRUTH = SOFT MAPS 
    global_step = 0

    # 6. Begin training
    for epoch in range(1, epochs+1):
        net.train()
        epoch_loss = 0
        with tqdm(total=n_train, desc=f'Epoch {epoch}/{epochs}', unit='img') as pbar:
            for batch in train_loader:
                images = batch['image']
                if useflow: 
                    # Add optical flow to input 
                    opticalflows = batch['flow']
                    images = torch.cat((images, opticalflows), dim=1)
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
                    if useatt: 
                        masks_pred = net(images, attention_maps)
                    else: 
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

                # optimizer.zero_grad(set_to_none=True)
                optimizer.zero_grad()
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
            net.eval()
            histograms = {}
            for tag, value in net.named_parameters():
                tag = tag.replace('/', '.')
                if not torch.isinf(value).any():
                    histograms['Weights/' + tag] = wandb.Histogram(value.data.cpu())
                if not torch.isinf(value.grad).any():
                    histograms['Gradients/' + tag] = wandb.Histogram(value.grad.data.cpu())

            val_score = evaluate(net, val_loader, device, useatt=useatt, addpos=addpositions, addflow=useflow)
            # scheduler.step(val_score)
            net.train()

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

        # 7. (Optional) Save checkpoint at each epoch 
        
        if save_checkpoint or save_best_checkpoint:
            adddirckp = 'U-Net-' + str(net.n_channels)
            if rgbtogs: 
                adddirckp += '-grayscale'
            if useflow: 
                adddirckp += '-w-flow'
            if addpositions: 
                adddirckp += '-w-positions'
            if useatt: 
                adddirckp += '-w-attention' 
            dirckp = dir_checkpoint / adddirckp
            dirckp.mkdir(parents=True, exist_ok=True)


        if save_checkpoint:            
            torch.save(net.state_dict(), str(dirckp / 'checkpoint_epoch{}.pth'.format(epoch)))
            logging.info(f'Checkpoint {epoch} saved!')
        
        # 8. (Optional) Save best model 
        if save_best_checkpoint: 
            if epoch == 1: 
                best_valscore = val_score 
                best_ckpt = 1 
                torch.save(net.state_dict(), str(dirckp / 'checkpoint_epoch_best.pth'))
                logging.info(f'Best checkpoint at {epoch} saved!')
                best_model_state = deepcopy(net.state_dict())
            else: 
                if val_score > best_valscore or n_val == 0: 
                    best_valscore = val_score
                    best_ckpt = epoch
                    torch.save(net.state_dict(), str(dirckp / 'checkpoint_epoch_best.pth'))
                    logging.info(f'Best checkpoint at {epoch} saved!')
                    best_model_state = deepcopy(net.state_dict())
            
            logging.info('Epoch loss: {}'.format(epoch_loss))

        # 9. Log all the previous parameters 
        experiment.log({
            'epoch': epoch, 
            'best epoch': best_ckpt, 
            'train loss epoch avg': epoch_loss, 
        })
    
    return best_ckpt, best_model_state

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
    parser.add_argument('--nosavebest', action='store_true', default=False, help="Don't save checkpoint of best epoch")
    parser.add_argument('--flow', action='store_true', default=False, help='Add optical flow to input')
    parser.add_argument('--pos', action='store_true', default=False, help='Add normalized position to input')
    parser.add_argument('--grayscale', '-gs', action='store_true', default=False, help='Convert RGB image to Greyscale for input')
    parser.add_argument('--test', action='store_true', default=False, help='Do the test after training')
    parser.add_argument('--viz', action='store_true', default=False, 
                        help='Visualize the images as they are processed')
    parser.add_argument('--foldnb', default=0, help='Number of the fold for cross-fold validation')
    return parser.parse_args()


if __name__ == '__main__':

    args = get_args()

    logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
    # Setting seed for reproducibility 
    set_seed(0)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logging.info(f'Using device {device}')
    
    # Change here to adapt to your data
    # n_channels=3 for RGB images
    # n_classes is the number of probabilities you want to get per pixel
    # if the model with attention is used, a different model will be loaded 
    n_channels = 3 
    if args.grayscale: 
        n_channels = 1 
    if args.pos: 
        n_channels += 2
    if args.flow: 
        n_channels += 2 
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
    # TRAINING SECTION 
    try:
        best_ckpt, best_model_state = train_net(
            net=net,
            epochs=args.epochs,
            batch_size=args.batch_size,
            learning_rate=args.lr,
            device=device,
            img_scale=args.scale,
            val_percent=args.val / 100,
            save_checkpoint=args.saveall,
            save_best_checkpoint=(not args.nosavebest),
            amp=args.amp, 
            useatt=args.attention, 
            useflow=args.flow, 
            lossframesdecay=args.framesdecay, 
            addpositions=args.pos, 
            rgbtogs=args.grayscale, 
            foldnumber=args.foldnb)
        logging.info(f'Best model is found at checkpoint #{best_ckpt}.')
    except KeyboardInterrupt:
        torch.save(net.state_dict(), 'INTERRUPTED.pth')
        logging.info('Saved interrupt')
        raise

    # TESTING SECTION     
    if args.test: 
        # Change here to adapt to your data
        # n_channels=3 for RGB images
        # n_classes is the number of probabilities you want to get per pixel
        # if the model with attention is used, a different model will be loaded 
        if args.attention: 
            testnet = UNetAtt(n_channels=n_channels, n_classes=args.classes, bilinear=args.bilinear)
        else: 
            testnet = UNet(n_channels=n_channels, n_classes=args.classes, bilinear=args.bilinear)
        testnet.load_state_dict(best_model_state, strict=False) # Recover the best model state, the one we usually keep 
        testnet.to(device=device)
        logging.info(f'Start testing... ')
        test_DICE = test_net(
        testnet, 
        device=device,
        images_dir=dir_img_test, 
        masks_dir=dir_mask_test, 
        attmaps_dir=dir_attmap_test, 
        flows_dir=dir_flow_test, 
        img_scale=args.scale,
        mask_threshold=0.5, 
        useatt=args.attention, 
        useflow=args.flow, 
        addpositions=args.pos, 
        rgbtogs=args.grayscale, 
        savepred=False, 
        visualize=args.viz)
        logging.info(f'DICE score of testing dataset is: {test_DICE}')
        