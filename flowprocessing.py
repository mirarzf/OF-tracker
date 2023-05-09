import argparse 
import logging
from pathlib import Path

import numpy as np

# CHOOSE INPUT DIRECTORIES 
## Optical flow input 
flowdir = Path("./data/test/flows")

## Folder where to save the predicted segmentation masks 
outdir = Path('./data/test/processed_flows')

def processflow(
        flowfilenames: list, 
        outputfilenames: list
) : 
    for input, output in zip(flowfilenames, outputfilenames): 
        flow = np.load(input)
        norms = np.sqrt(flow[:,:,0]**2+np.flow[:,:,1]**2) 
        normalized_x = flow[:,:,0]/norms
        normalized_y = flow[:,:,1]/norms
        norms = norms[np.newaxis, :]
        normalized_x = normalized_x[np.newaxis, :]
        normalized_y = normalized_y[np.newaxis, :]
        np.save(output, np.concatenate((normalized_x, normalized_y, norms), axis=0))
        logging.info(f"Processed optical flow saved to {output}")
        

def get_args():
    parser = argparse.ArgumentParser(description='Predict masks from input images')
    inputgroup = parser.add_mutually_exclusive_group(required=True)
    inputgroup.add_argument('--input', '-i', metavar='INPUT', nargs='+', help='Filenames of input images')
    inputgroup.add_argument('--dir', action='store_true', default=False, help='Use directories specified in flowprocessing.py file instead')
    parser.add_argument('--output', '-o', metavar='OUTPUT DIRECTORY', default=Path('./results/processed-flows'), help='Output directory for the results')
    return parser.parse_args()

def get_flow_filenames(args): 
    if args.dir: 
        return [f for f in flowdir.glob('*.png') if f.is_file()]
    else: 
        return args.input

def get_output_filenames(args): 
    if args.dir: 
        return [outdir / f for f in flowdir.glob('*.png') if f.is_file()]
    else: 
        return [args.output / f for f in flowdir.glob('*.png') if f.is_file()]

if __name__ == '__main__':    
    args = get_args() 
    logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')

    # Prepare the input files 
    flowfilenames = get_flow_filenames(args) 

    # Prepare the list of output file names 
    outputfilenames = get_output_filenames(args)
    
    processflow(
        flowfilenames=flowfilenames, 
        outputfilenames=outputfilenames
        )