from PIL import Image
from unet.unetutils.data_loading import AttentionDataset
import numpy as np
import albumentations as A 
import matplotlib.pyplot as plt 


# img = Image.open('./data/imgs/gg4541_4629_extract_1.png')
# img.show()
# mask = Image.open('./data/masks/gg4541_4629_extract_1.png')
# # mask.show()
# mask2 = Image.open('./data/masks/gg4541_4629_extract_2.png')
# # mask2.show()
# img2 = Image.open('./data/imgs/gg4541_4629_extract_1188.png')
# # img2.show()

# img = np.array(img)
# mask = np.array(mask)
# mask2 = np.array(mask2)
# print(img.shape)
# print(type(img),type(mask),type(mask2))

# geotransform = A.HorizontalFlip(p=1)
# geotransformtarg = A.Compose([geotransform], additional_targets={'attmap':'mask'})
# # transformed = geotransformtarg(image=img, mask=mask)
# print("yahoi")
# transformed = geotransformtarg(image=img, mask=mask, attmap=mask2)
# print(type(transformed))
# print(transformed.keys())
# img = Image.fromarray(transformed['image'])
# mask = Image.fromarray(transformed['mask'])
# mask2 = Image.fromarray(transformed['attmap'])
# img.show()
# # mask.show()
# # mask2.show()
# # print(transformed['masks'])

# colortrans = A.RandomBrightnessContrast(p=1)
# transformed = colortrans(image=np.asarray(img))
# img = Image.fromarray(transformed['image'])
# img.show()

# A = np.ones((3,3))
# B = 2*A
# print(A)
# print(B)
# print(A*B)

# import wandb
# api = wandb.Api()
# # run = api.run("mirarzf/OF-Tracker/runs/1egwkl65/")
# # run.config["use_positions"] = False
# # run.update()
# runs = api.runs(path="mirarzf/OF-Tracker")
# for run in runs: 
#     run.config["augmented_data"] = True
#     run.update()

import numpy as np 
import torch
import torch.nn.functional as F
a = -0.265468*np.ones((2,2,3))
a[0,0,0] = 1
a[1,1,2] = 2
a[0,0,1] = 1
a[1,0,1] = 1
a[0,1,0] = 0.1554
print(a)
a = torch.from_numpy(a)
print(F.softmax(a,dim=1).argmax(dim=1))
print(a.argmax(dim=1))
print(a.argmax(dim=0))
print(np.argmax(a.numpy(), axis=0))