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
b = torch.tensor([[[0, 0.5], [2, 4]]])
print(b)
print(b.size())

a = F.softmax(b.float(), dim=1)
print(a)
import wandb 
img = Image.fromarray(255*a.numpy()[0])
img.show()