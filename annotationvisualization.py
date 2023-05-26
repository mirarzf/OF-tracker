import os
import argparse

import numpy as np 
import pandas as pd
import cv2 as cv
import torch

from utils import annot_viz

from raft.raft import RAFT
from raft.raftutils import flow_viz
from raft.raftutils.utils import InputPadder

# torch.cuda.empty_cache()

### Folders 
outputfolder = ".\\results"
model_folder = "C:\\Users\\hvrl\\Documents\\RAFT-master\\models"

### Main code 
DEVICE = 'cuda'

def showAnnotatedPointsFlow(args): 

    ## SET THE ARGUMENTS FROM THE PARSER 
    annotatedpoints = args.dataset
    # annotatedpoints = "centerpointstest.csv"

    video_folder = args.videofolder

    scale = args.scale

    ## DATA PREPARATION 
    # Read the annotated data 
    apdf = pd.read_csv(annotatedpoints)

    # Get all the video_ids 
    video_id_list = apdf["video_id"].unique()

    ## OPTICAL FLOW MODEL PREPARATION 
    # Initialize RAFT model 
    model = torch.nn.DataParallel(RAFT(args))
    model.load_state_dict(torch.load(os.path.join(model_folder,args.model)))

    model = model.module
    model.to(DEVICE)
    model.eval()

    ## WORK ON EACH VIDEO
    with torch.no_grad(): 

        # Work on one video_id at a time 
        for video_id in video_id_list: 
            partdf = apdf[apdf["video_id"] == video_id]

            # Read original videos for parameters of the writer 
            origvidpath = os.path.join(video_folder, video_id + "_extract.mp4")
            cap = cv.VideoCapture(origvidpath)

            # Define output video parameters 
            outputname = "out_annotated_" + video_id + ".mp4" 
            print(outputname)
            fourcc = cv.VideoWriter_fourcc(*'mp4v')
            fps = cap.get(cv.CAP_PROP_FPS)
            # Initialize the padder for later and give the correct width and height 
            frame_width = int(scale*cap.get(cv.CAP_PROP_FRAME_WIDTH)) 
            frame_height = int(scale*cap.get(cv.CAP_PROP_FRAME_HEIGHT)) 
            ret, firstframe = cap.read()
            cap.release()
            firstframe = annot_viz.load_image(firstframe, (frame_width, frame_height))
            print("ffshape", firstframe.shape)
            padder = InputPadder((firstframe.shape))
            print(padder._pad)
            frame_width = frame_width + padder._pad[0] + padder._pad[1] 
            frame_height = frame_height + padder._pad[2] + padder._pad[3] 

            print(origvidpath, fps, frame_width, frame_height)

            # Define output video writer 
            output = cv.VideoWriter(os.path.join(outputfolder, outputname), fourcc, fps, (frame_width*2, frame_height))

             # Get the coordinates of the annotated points 
            aplist = []
            for index, row in partdf.iterrows(): 
                j, i = row["x_coord"], row["y_coord"]
                j, i = int(j*scale), int(i*scale)
                aplist.append((i,j))

            # Create random colors list for the annotated points visualization 
            n = len(aplist)
            randomcolors = [np.random.randint(256, size=3) for index in range(n)]
            print("# annotated points in video {video_id} : {n}".format(n=n, video_id = video_id))

            # Capture the first two frames
            cap = cv.VideoCapture(origvidpath)
            ret, beforeframe = cap.read() 
            ret, currentframe = cap.read()

            while ret and len(aplist) > 0: 
                # Prep the frames for optical flow retrieving
                beforeframe = annot_viz.load_image(beforeframe, (frame_width, frame_height))
                currentframe = annot_viz.load_image(currentframe, (frame_width, frame_height))
                padder = InputPadder(beforeframe.shape)
                beforeframe, currentframe = padder.pad(beforeframe, currentframe)

                # Retrieve the optical flow between beforeframe and currentframe 
                coordsubstract, currentflow = model(beforeframe, currentframe, iters=20, test_mode=True)
                
                # Draw on the image 
                currentframe = currentframe[0].permute(1,2,0).cpu().numpy()
                currentflow = currentflow[0].permute(1,2,0).cpu().detach().numpy()
                unscaledaplist = annot_viz.unscaledCoordlist(aplist, scale)
                frameimg = annot_viz.visualizePoint(currentframe.astype('uint8').copy() , unscaledaplist, color=randomcolors, scale=scale) # multiple points 
                flowimg = flow_viz.flow_to_image(currentflow, convert_to_bgr=True)
                flowimg = annot_viz.visualizePoint(flowimg, unscaledaplist, color=randomcolors, scale=scale) # multiple points 
                concatenation = cv.hconcat([frameimg, flowimg])
                # cv.imshow("result", concatenation)
                # cv.waitKey(0)
                output.write(concatenation)

                # Get the new vector of comparison 
                aplist, inFrameList = annot_viz.calculateNewPosition(aplist, currentflow) # multiple points 
                # Update the color list to only keep the points in frame 
                updaterandomcolors = []
                for bool, color in zip(inFrameList, randomcolors): 
                    if bool: 
                        updaterandomcolors.append(color)
                randomcolors = updaterandomcolors
                
                # print("Points etant encore in frame :", len(aplist), len(randomcolors))

                # Set new currentframe 
                beforeframe = currentframe
                ret, currentframe = cap.read()
        
            cap.release()
            output.release() 

        print(video_id, len(aplist))

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=
    "This script creates a video to visualize where the first annotated points are and the corresponding pseudo labels "
    "deduced by the optical flow vectors" )
    parser.add_argument('--model', default='raft-sintel.pth', help="restore checkpoint")
    parser.add_argument('--dataset', default='C:\\Users\\hvrl\\Documents\\data\\KU\\centerpoints.csv', help="CSV file with annotated points")
    parser.add_argument('--videofolder', '-vf', default='C:\\Users\\hvrl\\Documents\\data\\KU\\videos', help="folder containig the annotated videos")
    parser.add_argument('--scale', default=0.5, type=float, help="scale to resize the video frames. Default: 0.5")
    parser.add_argument('--small', action='store_true', help='use small model')
    parser.add_argument('--mixed_precision', action='store_true', help='use mixed precision')
    parser.add_argument('--alternate_corr', action='store_true', help='use efficent correlation implementation')
    args = parser.parse_args()

    showAnnotatedPointsFlow(args)