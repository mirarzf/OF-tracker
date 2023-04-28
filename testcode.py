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

flow_array = np.load(Path('data/flows/0838_0917_extract_1.npy'))

print(flow_array[:5,:5])
print(np.min(flow_array), np.max(flow_array))
flox = flow_array[:,:,0]
floy = flow_array[:,:,1]
norms = 2*np.sqrt(flox**2+floy**2)
norms = norms[:,:,np.newaxis]
norms = np.concatenate((norms, norms), axis=2)
result = flow_array/norms+np.ones(norms.shape)/2
print(np.min(result[0]), np.max(result[0]))
print(np.min(result[1]), np.max(result[1]))