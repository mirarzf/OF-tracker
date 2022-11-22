import cv2 as cv 
import numpy as np

import os 
from pathlib import Path

dir = Path('data/attmaps')
name = 's2_hotdog_0000000780'
print(list(dir.glob(name + '*.*')))