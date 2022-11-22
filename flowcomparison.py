# import sys
# sys.path.append('raft')

import os 
import argparse

import pandas as pd 

import numpy as np 
import cv2 as cv 
import torch

from utils import annot_viz

from raft.raft import RAFT
from raft.raftutils.utils import InputPadder


### Folders 
outputfolder = ".\\results"
model_folder = "C:\\Users\\hvrl\\Documents\\RAFT-master\\models"

### Main code 
DEVICE = 'cuda'

def compareFlowsToAnnotatedFlow(apcoord, flow): 
    '''
    In: 
    apcoord: (x,y) coordinates of the annotated point. We compare the optical flow 
    in the complete image to the optical flow at this point. 
    flow: numpy array of shape (height, width, 2) corresponding to the optical flow 

    Out: 
    compdot: numpy array of shape (height, width) with values between 0 and 1 
    '''

    norms = np.sqrt(flow[:,:,0]**2 + flow[:,:,1]**2)
    normapflow = norms[apcoord[0], apcoord[1]]

    apunitmat = flow[apcoord[0], apcoord[1],:]*np.ones(flow.shape)

    compdot = np.sum(apunitmat*flow, axis=2)/norms
    compdot /= normapflow
    # print(compdot.shape)
    # print("compdot result check : ", compdot[apcoord[0], apcoord[1]])

    # If the calculations are correct, values should be between 1 and -1. 
    # But because of the approximations, the maximum and minimum values found in the 
    # comparison matrix can be higher. 
    mincompdot = compdot.min()
    if mincompdot > -1: 
        mincompdot = -1 
    maxcompdot = compdot.max()
    if maxcompdot < 1: 
        maxcompdot = 1
    actualrange = maxcompdot - mincompdot 
    
    compdot = compdot - mincompdot*np.ones(compdot.shape)
    compdot /= actualrange
    # print(mincompdot, maxcompdot, actualrange)
    # print("compdot result check : ", compdot[apcoord[0], apcoord[1]])

    return compdot 

def compareFlowsToMultipleAnnotatedFlows(apcoordlist, flow): 
    '''
    In: 
    apcoordlist: list of (x,y) (pseudo-)annotated points to track. 
    flow: vector field repsenting the optical flow. 

    Out: 
    compdotsum: numpy array of shape (height, width) with values between 0 and 1, 
    result of the weighted sum of the comparison maps of the optical flow field with each annotated point flow vector. 
    '''

    compdotreslist = []
    for coord in apcoordlist: 
        compdotreslist.append(compareFlowsToAnnotatedFlow(coord, flow))
    compdotsum = np.sum(np.array(compdotreslist), axis=0)/len(apcoordlist)

    return compdotsum

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
            outputname = "out_sense_" + video_id + ".mp4" 
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
            print(padder._pad)
            frame_width = frame_width_old + padder._pad[0] + padder._pad[1] 
            frame_height = frame_height_old + padder._pad[2] + padder._pad[3] 

            print(origvidpath, fps, frame_width, frame_height)

            # Define output video writer 
            output = cv.VideoWriter(os.path.join(outputfolder, outputname), fourcc, fps, (frame_width*2, frame_height))

            # Get the coordinates of the annotated points 
            aplist = []
            for index, row in partdf.iterrows(): 
                j, i = row["x_coord"], row["y_coord"]
                j, i = int(j*scale)+padder._pad[0], int(i*scale)+padder._pad[2]
                aplist.append((i,j))

            # Create random colors list for the annotated points visualization 
            n = len(aplist)
            print(aplist)
            randomcolors = [np.random.randint(256, size=3) for index in range(n)]
            print("# annotated points in video {video_id} : {n}".format(n=n, video_id = video_id))

            # Capture the first two frames
            cap = cv.VideoCapture(origvidpath)
            ret, beforeframe = cap.read() 
            ret, currentframeimg = cap.read()

            # Prep the before frame and set the InputPadder 
            beforeframe = annot_viz.load_image(beforeframe, (frame_width, frame_height))
            padder = InputPadder(beforeframe.shape)
            beforeframe = padder.pad(beforeframe)[0]

            while ret and len(aplist) > 0: 
                # Prep the current frame for optical flow retrieving
                currentframeimg = annot_viz.load_image(currentframeimg, (frame_width, frame_height))
                currentframe = padder.pad(currentframeimg)[0]
                currentframeimg = currentframeimg[0].permute(1,2,0).cpu().numpy().astype('uint8')

                # Retrieve the optical flow between beforeframe and currentframe 
                flow_low, currentflow = model(beforeframe, currentframe, iters=20, test_mode=True)
                currentflow = currentflow[0].permute(1,2,0).cpu().detach().numpy()

                # Compare the "annotated" optical flow to all the other optical flow vectors 
                compres = compareFlowsToMultipleAnnotatedFlows(aplist, currentflow) 

                # Apply a threshold so we know which part of the image moves like the annotated point 
                seuil = np.quantile(compres, 0.9)
                # seuil = 0.9
                print("Le seuil est : ", seuil)
                # compres = np.where(compres < seuil, 0, compres)
                compres *= 255
                compres = np.expand_dims(compres, axis = -1)
                compres = np.concatenate((compres, compres, compres), axis = -1)
                compres = np.uint8(compres)
                compres = annot_viz.visualizePoint(compres, aplist, color=randomcolors)
                frameimg = annot_viz.visualizePoint(currentframeimg, aplist, color=randomcolors)
                concatenation = cv.hconcat([frameimg, compres])
                cv.imshow("output", compres)
                cv.waitKey(10)
                output.write(concatenation)

                # Get the new vector of comparison 
                aplist, inFrameList = annot_viz.calculateNewPosition(aplist, currentflow)
                # Update the color list to only keep the points in frame 
                updaterandomcolors = []
                for bool, color in zip(inFrameList, randomcolors): 
                    if bool: 
                        updaterandomcolors.append(color)
                randomcolors = updaterandomcolors
                
                print("Points etant encore in frame :", len(aplist), len(randomcolors))

                # Set new currentframe 
                beforeframe = currentframe
                ret, currentframeimg = cap.read()
        
            cap.release()
            output.release() 

        print(video_id, len(aplist))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="This script creates a video containing all the resulting map of attention "
                                                "to focus the segmentation map around the hand")
    parser.add_argument('--model', default='raft-things.pth', help="restore checkpoint")
    parser.add_argument('--dataset', default='C:\\Users\\hvrl\\Documents\\data\\KU\\centerpoints.csv', help="CSV file with annotated points")
    parser.add_argument('--videofolder', '-vf', default='C:\\Users\\hvrl\\Documents\\data\\KU\\videos', help="folder containig the annotated videos")
    parser.add_argument('--scale', default=0.5, type=float, help="scale to resize the video frames. Default: 0.5")
    parser.add_argument('--small', action='store_true', help='use small model')
    parser.add_argument('--mixed_precision', action='store_true', help='use mixed precision')
    parser.add_argument('--alternate_corr', action='store_true', help='use efficent correlation implementation')
    args = parser.parse_args()

    compareToAnnotatedPointsFlow(args)