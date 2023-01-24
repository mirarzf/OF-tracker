import torch
from torchvision import transforms 
import torchvision.transforms.functional as F
from PIL import Image
from unet.unetutils.data_augmentation import KUTransform


img = Image.open('./data/imgs/gg4541_4629_extract_1.png')
img.show()
transforms = KUTransform()
img = transforms(img)
img.show()
