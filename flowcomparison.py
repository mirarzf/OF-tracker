# import sys
# sys.path.append('raft')

import os 
import argparse

import pandas as pd 

import numpy as np 
import cv2 as cv 
import torch

from utils import annot_viz, flow_comp

from raft.raft import RAFT
from raft.raftutils.utils import InputPadder


### Folders 
outputfolder = ".\\results"
model_folder = "C:\\Users\\hvrl\\Documents\\RAFT-master\\models"

### Main code 
DEVICE = 'cuda'

def compareToAnnotatedPointsFlow(args): 

    ## SET THE ARGUMENTS FROM THE PARSER 
    annotatedpoints = args.dataset 
    annotatedpoints = "centerpointstest.csv"

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
            outputname = video_id + "_extract.mp4" 
            print(outputname)
            fourcc = cv.VideoWriter_fourcc(*'mp4v')
            fps = cap.get(cv.CAP_PROP_FPS)
            # Initialize the padder for later and give the correct width and height 
            frame_width_old = int(scale*cap.get(cv.CAP_PROP_FRAME_WIDTH)) 
            frame_height_old = int(scale*cap.get(cv.CAP_PROP_FRAME_HEIGHT)) 
            ret, firstframe = cap.read()
            cap.release()
            firstframe = annot_viz.load_image(firstframe, (frame_width_old, frame_height_old))
            print("ffshape", firstframe.shape)
            padder = InputPadder((firstframe.shape))
            # print(padder._pad)
            frame_width = frame_width_old + padder._pad[0] + padder._pad[1] 
            frame_height = frame_height_old + padder._pad[2] + padder._pad[3] 

            print(origvidpath, fps, frame_width, frame_height)

            # Define output video writer 
            output = cv.VideoWriter(os.path.join(outputfolder, outputname), fourcc, fps, (frame_width, frame_height))

            # Get the coordinates of the annotated points and their type 
            # 0 : background 
            # 1 : left hand 
            # 2 : right hand 
            aplist = []
            for index, row in partdf.iterrows(): 
                j, i, type = row["x_coord"], row["y_coord"], row["type"]
                j, i = int(j*scale)+padder._pad[0], int(i*scale)+padder._pad[2]
                aplist.append((i,j,type))

            # Check number of points that are annotated
            n = len(aplist)
            print(aplist)
            print("# annotated points in video {video_id} : {n}".format(n=n, video_id = video_id))

            # Capture the first two frames
            cap = cv.VideoCapture(origvidpath)
            ret, beforeframe = cap.read() 
            ret, currentframeimg = cap.read()

            # Prep the before frame and set the InputPadder 
            beforeframe = annot_viz.load_image(beforeframe, (frame_width, frame_height))
            padder = InputPadder(beforeframe.shape)
            beforeframe = padder.pad(beforeframe)[0]

            while ret:      
                # Prep the current frame for optical flow retrieving
                currentframeimg = annot_viz.load_image(currentframeimg, (frame_width, frame_height))
                currentframe = padder.pad(currentframeimg)[0]
                currentframeimg = currentframeimg[0].permute(1,2,0).cpu().numpy().astype('uint8')

                if len(aplist) > 0:         
                    # Retrieve the optical flow between beforeframe and currentframe 
                    coordsubstract, currentflow = model(beforeframe, currentframe, iters=20, test_mode=True)
                    currentflow = currentflow[0].permute(1,2,0).cpu().detach().numpy()

                    # Compare the "annotated" optical flow to all the other optical flow vectors 
                    compres = flow_comp.compareFlowsToMultipleAnnotatedFlows(aplist, currentflow) 
                else: 
                    compres = np.zeros(currentframeimg.shape[:2])

                # Apply a threshold so we know which part of the image moves like the annotated points 
                seuil = np.quantile(compres, 0.75)
                # seuil = 0.9
                print("Le seuil est : ", seuil)
                # compres = np.where(compres < seuil, 0, compres)
                compres *= 255
                compres = np.expand_dims(compres, axis = -1)
                compres = np.concatenate((compres, compres, compres), axis = -1)
                compres = np.uint8(compres)
                cv.imshow("output", compres)
                cv.waitKey(1)
                output.write(compres)

                # Get the new vector of comparison 
                aplist, inFrameList = annot_viz.calculateNewPosition(aplist, currentflow)
                
                # print("Points etant encore in frame :", len(aplist))

                # Set new currentframe 
                beforeframe = currentframe
                ret, currentframeimg = cap.read()
        
            cap.release()
            output.release() 

        print(f"Points etant encore in frame dans la video {video_id} : {len(aplist)}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="This script creates a video containing all the resulting map of attention "
                                                "to focus the segmentation map around the hand")
    parser.add_argument('--model', default='raft-things.pth', help="restore checkpoint")
    parser.add_argument('--dataset', default='C:\\Users\\hvrl\\Documents\\data\\KU\\centerpoints.csv', help="CSV file with annotated points")
    parser.add_argument('--videofolder', '-vf', default='C:\\Users\\hvrl\\Documents\\data\\KU\\videos\\annotated', help="folder containig the annotated videos")
    parser.add_argument('--scale', default=0.5, type=float, help="scale to resize the video frames. Default: 0.5")
    parser.add_argument('--small', action='store_true', help='use small model')
    parser.add_argument('--mixed_precision', action='store_true', help='use mixed precision')
    parser.add_argument('--alternate_corr', action='store_true', help='use efficent correlation implementation')
    args = parser.parse_args()

    compareToAnnotatedPointsFlow(args)