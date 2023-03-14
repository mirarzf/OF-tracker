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
n = 10
data_indices = list(range(n))
np.random.shuffle(data_indices)
# print(data_indices)
k_number = 4 
last_idx_of_split = []
len_split = n // k_number + 1 
last_idx_of_split = list(range(len_split-1, n, len_split))
print("short lengths of split:", n//k_number)

last_idx_of_split = []
# for i in range(n%nb_split): 
#     last_idx_of_split.append((i+1)*n // nb_split+1)
#     print(last_idx_of_split[-1], "aha")
# for i in range(last_idx_of_split[-1]+n//nb_split, n, n//nb_split): 
#     last_idx_of_split.append(i)
#     print(last_idx_of_split[-1], "oho")
q = n // k_number 
r = n%k_number
for i in range(k_number+1): 
    if i <= r: 
        last_idx_of_split.append(i*(q+1))
        print(last_idx_of_split[-1], i, "aha")
    else: 
        last_idx_of_split.append((i+1)*q)
        print(last_idx_of_split[-1], i, "oho")
last_idx_of_split.append(n)
# print(split_indices, "splish splish")
# print("les splits : ", [data_indices[last_idx_of_split[foldnb]:last_idx_of_split[foldnb+1]] for foldnb in range(nb_split)])
# print("last_idx_of_split", last_idx_of_split)

# foldnb = 2
# val_files = [imgfilenames[idx] for idx in data_indices[last_idx_of_split[foldnb]:last_idx_of_split[foldnb+1]]] 
# train_files = [imgfilenames[idx] for idx in data_indices[:last_idx_of_split[foldnb]]+data_indices[last_idx_of_split[foldnb+1]:]] 
# print(len(val_files))
# print(data_indices[last_idx_of_split[foldnb]:last_idx_of_split[foldnb+1]])
# print(len(train_files))
# print(val_files)
foldnb = 2
print(f"Il y a {n} fichiers. \
      Ils sont regroupes comme ceci: {data_indices}. \n\
      Les ids du val dataset sont : \n {[data_indices[last_idx_of_split[foldnb]:last_idx_of_split[foldnb+1]]]} \n\
      Ils sont repartis de la maniere suivante : \n {[data_indices[last_idx_of_split[split]:last_idx_of_split[split+1]] for split in range(k_number)]}")