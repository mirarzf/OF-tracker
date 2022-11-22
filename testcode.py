import cv2 as cv 
import numpy as np

import os 
from pathlib import Path

import torch 

x = torch.tensor([[[1, 2, 3], [4, 5, 6]]])
print(x.size(), '\n', x)
y = x.repeat(4, 1, 1)
print(y.size())
print(y.size()[0])
print(y)