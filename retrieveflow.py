# This script is used to retrieve the optical flow and change the size of the input if needed. 
# The size of the input to be changed is changed with the destsize argument in the parser. 

######################################################################################### 

# import sys
# sys.path.append('raft')

from pathlib import Path 
import argparse
import logging

import numpy as np 
import cv2 as cv 
import torch

from utils import annot_viz 

from raft.raft import RAFT
from raft.raftutils.utils import InputPadder
from raft.raftutils.flow_viz import flow_to_image


### Folders 
videofolder = 'C:\\Users\\hvrl\\Documents\\data\\KU\\videos\\annotated'
outputfolder = ".\\results\\flows\\KU"
model_folder = "C:\\Users\\hvrl\\Documents\\RAFT-master\\models"

def retrieveFlow(args): 

    video_folder = Path(args.videofolder)
    output_folder = Path(args.outputfolder)

    scale = args.scale

    # Get all the video_ids 
    video_id_list = [video_id for video_id in video_folder.iterdir() if video_id.is_file()]

    ## OPTICAL FLOW MODEL PREPARATION 
    # Initialize RAFT model 
    model = torch.nn.DataParallel(RAFT(args))
    model.load_state_dict(torch.load(Path(model_folder) / args.model))

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logging.info(f'Using device {device}')
    model = model.module
    model.to(device)
    model.eval()

    # Define output folders 
    if not output_folder.exists(): 
        output_folder.mkdir()
    
    if args.saveRGB: 
        if not (output_folder / 'RGB').exists(): 
            (output_folder / 'RGB').mkdir()

    if args.savevideo: 
        if not (output_folder / 'video').exists(): 
            (output_folder / 'video').mkdir()

    ## WORK ON EACH VIDEO OF THE FOLDER 
    with torch.no_grad(): 

        # Work on one video_id at a time 
        for video_id in video_id_list: 
            logging.info(f'Processing video {video_id.stem}... ')

            # Read original videos for frame scale and eventual output video writer 
            origvidpath = str(video_id)
            cap = cv.VideoCapture(origvidpath)

            # Initialize the padder for later and give the correct width and height 
            if args.width == None and args.height == None: 
                frame_width_old = int(scale*cap.get(cv.CAP_PROP_FRAME_WIDTH)) 
                frame_height_old = int(scale*cap.get(cv.CAP_PROP_FRAME_HEIGHT)) 
            else: 
                frame_width_old = args.width 
                frame_height_old = args.height 

            ret, firstframe = cap.read()
            cap.release()
            firstframe = annot_viz.load_image(firstframe, (frame_width_old, frame_height_old))
            padder = InputPadder((firstframe.shape))
            paddervalues = padder._pad
            
            frame_width = frame_width_old + padder._pad[0] + padder._pad[1] 
            frame_height = frame_height_old + padder._pad[2] + padder._pad[3] 
            if args.savevideo: 
                # Create video writer to save color coded optical flow to a video 
                ## Define output video parameters 
                outputname = video_id + "_extract.mp4" 
                fourcc = cv.VideoWriter_fourcc(*'mp4v')
                fps = cap.get(cv.CAP_PROP_FPS)

                logging.info(f'Creating new video writer:'
                             f'\toriginal video path: {origvidpath}'
                             f'\t{fps} frames per second'
                             f'\tframe width, frame_height = {frame_width_old}, {frame_height_old}')

                ## Define output video writer 
                outputvideo = output_folder / 'video' / outputname
                output = cv.VideoWriter(outputvideo, fourcc, fps, (frame_width_old, frame_height_old))

            for firstframenb in range(0, args.framestep): 
                # Capture the first two frames
                cap = cv.VideoCapture(origvidpath)
                ret, beforeframe = cap.read() 
                # Frame counter for output name 
                framecounter = firstframenb+1
                while framecounter % args.framestep != firstframenb: 
                    ret, currentframeimg = cap.read()
                    framecounter += 1 
                ret, currentframeimg = cap.read()

                # Prep the before frame and set the InputPadder 
                beforeframe = annot_viz.load_image(beforeframe, (frame_width, frame_height))
                padder = InputPadder(beforeframe.shape)
                beforeframe = padder.pad(beforeframe)[0]

                while ret:      
                    # Prep the current frame for optical flow retrieving
                    currentframeimg = annot_viz.load_image(currentframeimg, (frame_width, frame_height))
                    currentframe = padder.pad(currentframeimg)[0]

                    coordsubstract, currentflow = model(beforeframe, currentframe, iters=20, test_mode=True)
                    currentflow = currentflow[0].permute(1,2,0).cpu().detach().numpy()
                    currentflow = currentflow[paddervalues[2]:currentflow.shape[0]-paddervalues[3], paddervalues[0]:currentflow.shape[1]-paddervalues[1],:]

                    flowimg = flow_to_image(currentflow, args.clippercentage)

                    # Save optical flow 
                    # outputname = str(video_id.stem)[:-2].lower() + f'{framecounter:010}' # TO MATCH GTEA FRAMES NAMES 
                    outputname = str(video_id.stem) + f'_{framecounter}' # TO MATCH SURGERY VIDEO FRAMES NAMES 
                    ## Save optical flow Numpy array 
                    outputnpy = output_folder / (outputname + '.npy')
                    np.save(outputnpy, currentflow)
                    logging.info(f'Saved to {outputnpy}')

                    ## Save optical flow color coded RGB image 
                    if args.saveRGB and args.framestep == 1: 
                        outputimg = output_folder / 'RGB' / (outputname + '.jpg')
                        cv.imwrite(str(outputimg), flowimg)
                        logging.info(f'Saved to {outputimg}')

                    ## Save optical flow in a video 
                    if args.savevideo: 
                        output.write(flowimg)

                    # Set new currentframe 
                    beforeframe = currentframe
                    framecounter += 1 
                    while framecounter % args.framestep != firstframenb: 
                        ret, currentframeimg = cap.read()
                        framecounter += 1 
                    ret, currentframeimg = cap.read()
            
                cap.release()
            if args.savevideo and args.framestep == 1: 
                output.release() 
                logging.info(f'Video of optical flow saved to {outputvideo}')

def get_args(): 
    parser = argparse.ArgumentParser(description=
                                     "This script retrieves optical flow from the videos "
                                     "in the folder given in input")
    parser.add_argument('--model', default='raft-things.pth', help="restore checkpoint")
    parser.add_argument('--videofolder', '-vf', default=videofolder, help="folder containig the videos to extract optical flow from")
    parser.add_argument('--outputfolder', default=outputfolder, help="folder to save the optical flow frames to")
    parser.add_argument('--scale', default=0.5, type=float, help="scale to resize the video frames. Default: 0.5")
    parser.add_argument('--framestep', default=1, type=int, help="flow is computed between frames spaced by this numbder (framestep)")
    parser.add_argument('--width', type=int, help="width to resize the video frames to")
    parser.add_argument('--height', type=int, help="height to resize the video frames to")
    parser.add_argument('--clippercentage', '-cp', type=int, default="100", help="percentage of the size of the image to clip the optical flow value to")
    parser.add_argument('--small', action='store_true', help='use small model')
    parser.add_argument('--mixed_precision', action='store_true', help='use mixed precision')
    parser.add_argument('--alternate_corr', action='store_true', help='use efficent correlation implementation')
    parser.add_argument('--saveRGB', action='store_true', default=False, help='Save color coded optical flow')
    parser.add_argument('--savevideo', action='store_true', default=False, help='Save color coded optical flow into a video')
    return parser.parse_args()

if __name__ == '__main__':
    args = get_args()

    logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')

    retrieveFlow(args)