import os 
from pathlib import Path
import subprocess
import logging 

# CHOOSE INPUT DIRECTORIES 
imgdir = Path("../data/GTEA/frames")
attmapdir = None # Path("./")
outdir = Path("./results")

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logging.info(f'The directory for input images is {imgdir}. ')
if attmapdir != None: 
    logging.info(f'The directory for input attention maps is {attmapdir}. ')
logging.info(f'The directory for output prediction maps is {outdir}. ')

# SORT FILENAMES FOR INPUT 
imgfilenames = [os.path.join(imgdir, f) for f in os.listdir(imgdir) if "s4" in f]
imgfilenames.sort()

if attmapdir != None: 
    attmapnames = [os.path.join(attmapdir, os.path.basename(imgfilename)) for imgfilename in imgfilenames]
    attmapnames.sort()

logging.info(f'There are {len(imgfilenames)} input images. ')

# START THE COMMAND 
if attmapdir != None: 
    logging.info(f'The UNet model with attention is being used. ')
else: 
    logging.info(f'The standard UNet model is being used. ')

ckp = Path("./checkpoints/tGTEA_bs16_e10.pth")
cmd = ["python", "predict.py", "--model", ckp]
logging.info(f'The model loads the weights from {ckp}. ')

# ADD INPUT IMAGES 
cmd.append('-i')
cmd += imgfilenames

# ADD INPUT ATTENTION MAPS (IF PRESENT IN INPUT)
if attmapdir != None: 
    cmd.append('-a')
    cmd += [os.path.join(outdir, os.path.basename(imgfilename)) for imgfilename in imgfilenames]

# ADD OUTPUT FILENAMES 
cmd.append('-o')
cmd += [os.path.join(outdir, os.path.basename(imgfilename)) for imgfilename in imgfilenames]

# RUN THE COMMAND 
subprocess.run(cmd)

logging.info(f'Prediction complete! ')