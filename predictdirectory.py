import os 
from pathlib import Path
import subprocess

# imgdir = Path("../data/KU/frames")
# maskdir = Path("../data/KU/masks")

imgdir = Path("../data/GTEA/frames")

# imgfilenames = [os.path.join(imgdir, f) for f in os.listdir(maskdir) if "mirrored" not in f]
# maskfilenames = [os.path.join(maskdir, f) for f in os.listdir(maskdir) if "mirrored" not in f]


imgfilenames = [os.path.join(imgdir, f) for f in os.listdir(imgdir) if "s4" in f]

imgfilenames.sort()
# maskfilenames.sort()
# print(imgfilenames[:3], len(imgfilenames), len(maskfilenames))


os.environ['KMP_DUPLICATE_LIB_OK']='True'
n = len(imgfilenames)
for imgfilename in imgfilenames: 
    print(imgfilename)
    subprocess.run(["python", "predict.py", "--model", Path("./checkpoints/gtea_rgb_bs16_ckp5.pth"), "-i", imgfilename, '-o', os.path.join(Path("./results/"), os.path.basename(imgfilename))])