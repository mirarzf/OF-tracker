import torch
from torchvision import transforms as T 
import torchvision.transforms.functional as F
from PIL import Image
from unet.unetutils.data_augmentation import KUTransform, GeometricTransform


img = Image.open('./data/imgs/gg4541_4629_extract_1.png')
img.show()
img2 = Image.open('./data/imgs/gg4541_4629_extract_1188.png')
img2.show()
dataaugtransform = {'geometric': GeometricTransform, 
                    'color': KUTransform}
transform = dataaugtransform['geometric']()

tensorize = T.ToTensor()
pilize = T.ToPILImage()
dico = { 
    'img': img, 
    'mask': img2
}
dico = transform(dico)
img = dico['img']
img2 = dico['mask']
img.show()
img2.show()
print(img.size, img2.size)

# transform = T.RandomChoice([
#      T.RandomHorizontalFlip(), 
#      T.RandomVerticalFlip()
# ])
# # img = transform(img)
# # img2 = transform(img2)
# print("yahoi")
# img.show()
# print("img2")
# img2.show()
