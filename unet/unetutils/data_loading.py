import logging
from os import listdir
from os.path import splitext
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset


class AttentionDataset(Dataset): 
    def __init__(self, images_dir: str, masks_dir: str, scale: float = 1.0, mask_suffix: str = '', attmaps_dir: str = '', withatt: bool = True):
        self.withatt = withatt

        self.images_dir = Path(images_dir)
        self.masks_dir = Path(masks_dir)
        self.attmaps_dir = Path(attmaps_dir)
        assert 0 < scale <= 1, 'Scale must be between 0 and 1'
        self.scale = scale
        self.mask_suffix = mask_suffix

        self.ids = [splitext(file)[0] for file in listdir(images_dir) if not file.startswith('.')]
        if not self.ids:
            raise RuntimeError(f'No input file found in {images_dir}, make sure you put your images there')
        logging.info(f'Creating dataset with {len(self.ids)} examples')

    def __len__(self):
        return len(self.ids)

    @staticmethod
    def preprocess(pil_img, scale, is_mask):
        w, h = pil_img.size
        newW, newH = int(scale * w), int(scale * h)
        assert newW > 0 and newH > 0, 'Scale is too small, resized images would have no pixel'
        newW, newH = 572, 572
        # assert newW > w or newH > h, 'Input images will be upsampled due to one dimension of the image being smaller than 572'
        pil_img = pil_img.resize((newW, newH), resample=Image.NEAREST if is_mask else Image.BICUBIC)
        img_ndarray = np.asarray(pil_img)
        if is_mask: 
            img_ndarray = np.where(img_ndarray == 1 or img_ndarray == 2, 255, 0)

        if not is_mask:
            if img_ndarray.ndim == 2:
                img_ndarray = img_ndarray[np.newaxis, ...]
            else:
                img_ndarray = img_ndarray.transpose((2, 0, 1))

            img_ndarray = img_ndarray / 255

        return img_ndarray

    @staticmethod
    def load(filename):
        ext = splitext(filename)[1]
        if ext == '.npy':
            return Image.fromarray(np.load(filename))
        elif ext in ['.pt', '.pth']:
            return Image.fromarray(torch.load(filename).numpy())
        else:
            return Image.open(filename)

    def __getitem__(self, idx):
        name = self.ids[idx]
        mask_file = list(self.masks_dir.glob(name + self.mask_suffix + '.*'))
        img_file = list(self.images_dir.glob(name + '.*'))
        if self.withatt: 
            attmap_file = list(self.attmaps_dir.glob(name + '.*'))
        
        assert len(img_file) == 1, f'Either no image or multiple images found for the ID {name}: {img_file}'
        assert len(mask_file) == 1, f'Either no mask or multiple masks found for the ID {name}: {mask_file}.'
        if self.withatt: 
            assert len(attmap_file) == 1, f'Either no attention map or multiple attention maps found for the ID {name}: {attmap_file}.'
        
        mask = self.load(mask_file[0])
        img = self.load(img_file[0])
        if self.withatt: 
            attmap = self.load(attmap_file[0])

        if self.withatt: 
            assert img.size == mask.size and img.size == attmap.size, \
                f'Image, mask and attention map {name} should be the same size, but are {img.size}, {mask.size} and {attmap.size}'
        else: 
            assert img.size == mask.size, \
                f'Image and mask {name} should be the same size, but are {img.size} and {mask.size}'

        img = self.preprocess(img, self.scale, is_mask=False)
        mask = self.preprocess(mask, self.scale, is_mask=True)
        if self.withatt: 
            attmap = self.preprocess(attmap, self.scale, is_mask=False)
        
        retdict = {}
        retdict['image'] = torch.as_tensor(img.copy()).float().contiguous()
        retdict['mask'] = torch.as_tensor(mask.copy()).long().contiguous()
        if self.withatt: 
            retdict['attmap'] = torch.as_tensor(attmap.copy()).float().contiguous()

        return retdict

class BasicDataset(AttentionDataset): 
    def __init__(self, images_dir: str, masks_dir: str, scale: float = 1, mask_suffix: str = '', attmaps_dir: str = '', withatt: bool = True):
        super().__init__(images_dir, masks_dir, scale, mask_suffix, attmaps_dir, withatt=False)


class CarvanaDataset(BasicDataset):
    def __init__(self, images_dir, masks_dir, scale=1):
        super().__init__(images_dir, masks_dir, scale, mask_suffix='_mask')