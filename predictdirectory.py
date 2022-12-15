import os 
from pathlib import Path
import subprocess
import logging 
import copy

# CHOOSE INPUT DIRECTORIES 
# imgdir = Path("../data/GTEA/frames")
imgdir = Path("./data/imgs")
# attmapdir = None # Path("./")
attmapdir = Path("./data/attmaps")
outdir = Path("./results/unet")

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logging.info(f'The directory for input images is {imgdir}. ')
if attmapdir != None: 
    logging.info(f'The directory for input attention maps is {attmapdir}. ')
logging.info(f'The directory for output prediction maps is {outdir}. ')

# SORT FILENAMES FOR INPUT 
imgfilenames = [os.path.join(imgdir, f) for f in os.listdir(imgdir) if ".png" in f]
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

ckp = Path("./checkpoints/woattention/ptGTEA_tKU_bs1_e5.pth")
cmdbegin = ["python", "predict.py", "--model", ckp]
logging.info(f'The model loads the weights from {ckp}. ')
cmd = cmdbegin.copy()

cmdimg = []
cmdatt = []
cmdout = []

for i, imgfilename in enumerate(imgfilenames): 
    cmdimg.append(imgfilename)
    if attmapdir != None: 
        cmdatt.append(os.path.join(attmapdir, os.path.basename(imgfilename)))
    cmdout.append(os.path.join(outdir, os.path.basename(imgfilename)))

    if i%32==0 or len(imgfilenames)-i == 1: 

        # ADD INPUT IMAGE 
        cmd.append('-i')
        cmd += [e for e in cmdimg]

        # ADD INPUT ATTENTION MAP (IF PRESENT IN INPUT)
        if attmapdir != None: 
            cmd.append('-a')
            cmd += [e for e in cmdatt]

        # ADD OUTPUT FILENAME 
        cmd.append('-o')
        cmd += [e for e in cmdout]

        # RUN THE COMMAND 
        # print(cmd)
        subprocess.run(cmd)

        # EMPTY COMMAND TO RUN THE NEXT ONE AGAIN 
        cmd = cmdbegin.copy()
        cmdimg = []
        cmdatt = []
        cmdout = []

logging.info(f'Prediction complete! ')