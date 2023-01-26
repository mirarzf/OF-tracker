import torch
from torchvision import transforms as T 
import torchvision.transforms.functional as F
from PIL import Image
from unet.unetutils.data_loading import AttentionDataset
import numpy as np


img = Image.open('./data/imgs/gg4541_4629_extract_1.png')
# img.show()
img2 = Image.open('./data/imgs/gg4541_4629_extract_1188.png')
# img2.show()


img = torch.from_numpy(AttentionDataset.preprocess(img, 1, is_mask=False))
_, w, h = img.shape
x = torch.tensor(np.arange(h)/(h-1))
y = torch.tensor(np.arange(w)/(w-1))
grid_x, grid_y = torch.meshgrid(x, y, indexing='ij')
print(img.size(), grid_x.size(), grid_y.size())
grid_x = grid_x.repeat(1, 1, 1)
grid_y = grid_y.repeat(1, 1, 1)
print(grid_x)
print(img.size(), grid_x.size(), grid_y.size(), "LE CALME AVANT LA TEMPETE")
img = torch.cat((img, grid_x, grid_y), dim=0)
print(img.size())
print(img[:, 50, 50])