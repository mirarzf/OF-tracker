import numpy as np 
from pathlib import Path


# REPRODUCIBILITY 
import random
def set_seed(seed: int = 42) -> None:
    np.random.seed(seed)
    random.seed(seed)
    print(f"Random seed set as {seed}. \n")
# END REPRODUCIBILLITY 
set_seed(0)


dir_img = Path('./data/imgs/')
imgfilenames = [f for f in dir_img.iterdir() if f.is_file() and f.name != '.gitkeep'] 
# print(imgfilenames)

n = len(imgfilenames)
data_indices = np.arange(n)
np.random.shuffle(data_indices)
data_indices = list(data_indices)
# print(data_indices)
nb_split = 9
len_split = n // nb_split + 1 
split_indices = np.array_split(data_indices, nb_split)
last_idx_of_split = list(range(len_split-1, n, len_split))
print("short lengths of split:", n//nb_split)

last_idx_of_split = [0]
for i in range(n%nb_split): 
    last_idx_of_split.append((i+1)*n // nb_split+1)
    print(last_idx_of_split[-1], "aha")
for i in range(last_idx_of_split[-1]+n//nb_split, n, n//nb_split): 
    last_idx_of_split.append(i)
    print(last_idx_of_split[-1], "oho")
last_idx_of_split.append(n)
print(split_indices, "splish splish")
print([data_indices[last_idx_of_split[foldnb]:last_idx_of_split[foldnb+1]] for foldnb in range(nb_split)])
print(last_idx_of_split)

foldnb = 2
val_files = [imgfilenames[idx] for idx in data_indices[last_idx_of_split[foldnb]:last_idx_of_split[foldnb+1]]] 
train_files = [imgfilenames[idx] for idx in data_indices[:last_idx_of_split[foldnb]]+data_indices[last_idx_of_split[foldnb+1]:]] 
print(len(val_files))
print(data_indices[last_idx_of_split[foldnb]:last_idx_of_split[foldnb+1]])
print(len(train_files))
print(val_files)