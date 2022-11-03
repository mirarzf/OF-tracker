import sys
sys.path.append('raftcore')

import os
import glob
import argparse

import numpy as np 
import pandas as pd
import cv2 as cv
import torch
from PIL import Image

from utils import flow_viz, annot_viz

from raft import RAFT
from utils.utils import InputPadder

### Folders 

outputfolder = ".\\results"

### Main code 
DEVICE = 'cuda'

def showAnnotatedPointsFlow(args): 

    annotatedpoints = args.dataset
    annotatedpoints = "centerpointstest.csv"

    video_flow_folder = "C:\\Users\\hvrl\\Documents\\data\\KU\\of" 
    video_folder = "C:\\Users\\hvrl\\Documents\\data\\KU\\videos"
    model_folders = "C:\\Users\\hvrl\\Documents\\RAFT-master\\models"

    # Initialize RAFT model 
    model = torch.nn.DataParallel(RAFT(args))
    model.load_state_dict(torch.load(os.path.join(model_folders,args.model)))

    model = model.module
    model.to(DEVICE)
    model.eval()

    # Read the annotated data 
    apdf = pd.read_csv(annotatedpoints)

    # Get all the video_ids 
    video_id_list = apdf["video_id"].unique()

    # Work on one video_id at a time 
    for video_id in video_id_list: 
        partdf = apdf[apdf["video_id"] == video_id]

        # Get the coordinates of the annotated points 
        aplist = []
        for index, row in partdf.iterrows(): 
            j, i = row["x_coord"], row["y_coord"]
            aplist.append((i,j))

        n = len(aplist)
        randomcolors = [np.random.randint(256, size=3) for index in range(n)]
        print(randomcolors)

        # Read original videos for parameters of the writer 
        origvidpath = os.path.join(video_folder, video_id + "_extract.mp4")
        cap = cv.VideoCapture(origvidpath)

        # Define output video parameters 
        outputname = "out_annotated_" + video_id + ".mp4" 
        print(outputname)
        fourcc = cv.VideoWriter_fourcc(*'mp4v')
        fps = cap.get(cv.CAP_PROP_FPS)
        frame_width = currentflow.shape[1]
        frame_height = currentflow.shape[0]
        print(origvidpath, fps)

        # Define output video writer 
        output = cv.VideoWriter(os.path.join(outputfolder, outputname), fourcc, fps, (frame_width*2, frame_height))

        # Capture the first two frames 
        ret, frame1 = cap.read() 
        ret, frame2 = cap.read()

        # Calculate the optical flow for first pair of frames 
        
        framenames = [f for f in os.listdir(video_flow_folder) if video_id in f and "mirrored" not in f]
        framenames.sort()

        flowfiles = [os.path.join(video_flow_folder, f) for f in framenames]

        # Retrieve the optical flow of the first annotated frame 
        currentofname = os.path.join(video_flow_folder, partdf["video_frame_id"].iloc[0] + ".npy")
        currentframename = os.path.join(video_frames_folder, partdf["video_frame_id"].iloc[0] + ".png")
        currentflow = np.load(currentofname)
        currentframe = cv.imread(currentframename)
        
        # Compare for each frame in the video the optical flow of the annotated flow to the one of the annotated point 
        currentframenb = partdf["frame_number"].iloc[0]
        currentofname = partdf["video_frame_id"].iloc[0]
        while currentframenb < len(framenames) and len(aplist) > 0: 
            # Draw on the image 
            annot_viz.visualizePoint(currentframe, aplist, color=randomcolors) # multiple points 
            flowimg = flow_viz.flow_to_image(currentflow, convert_to_bgr=True)
            annot_viz.visualizePoint(flowimg, aplist, color=randomcolors) # multiple points 
            concatenation = cv.hconcat([currentframe, flowimg])
            output.write(concatenation)

            # Determine the next optical flow vector of comparison
            currentframenb += 1

            # Get the new frame of optical flow 
            newofnamelist = os.path.basename(os.path.splitext(currentofname)[0]).split("_")[:-1]
            newofnamelist.append(str(currentframenb))
            newofname = annot_viz.reconstructFilenameFromList(newofnamelist)
            currentofname = os.path.join(video_flow_folder, newofname + ".npy")
            currentframename = os.path.join(video_frames_folder, newofname + ".png")

            # Set new currentflow and new currentframe 
            currentflow = np.load(currentofname)
            currentframe = cv.imread(currentframename)

            # Get the new vector of comparison 
            aplist, inFrameList = annot_viz.calculateNewPosition(aplist, currentflow) # multiple point 
            # Update the color list to only keep the points in frame 
            updaterandomcolors = []
            for bool, color in zip(inFrameList, randomcolors): 
                if bool: 
                    updaterandomcolors.append(color)
            randomcolors = updaterandomcolors
            
            print("Points etant encore in frame :", len(aplist), len(randomcolors))
        
        cap.release()
        output.release() 

        print(video_id, len(aplist))

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', default='raft-sintel.pth', help="restore checkpoint")
    parser.add_argument('--dataset', default='C:\\Users\\hvrl\\Documents\\data\\KU\\centerpoints.csv', help="dataset for evaluation")
    parser.add_argument('--small', action='store_true', help='use small model')
    parser.add_argument('--mixed_precision', action='store_true', help='use mixed precision')
    parser.add_argument('--alternate_corr', action='store_true', help='use efficent correlation implementation')
    args = parser.parse_args()

    showAnnotatedPointsFlow(args)