import os 
import pandas as pd 

import numpy as np 
import matplotlib.pyplot as plt 
import cv2 as cv 

from utils import annot_viz

video_flow_folder = "C:\\Users\\hvrl\\Documents\\data\\KU\\of" 
video_masks_folder = "C:\\Users\\hvrl\\Documents\\data\\KU\\masks"
annotatedpoints = "C:\\Users\\hvrl\\Documents\\data\\KU\\centerpoints.csv"
video_folder = "C:\\Users\\hvrl\\Documents\\data\\KU\\videos"

outputfolder = ".\\results"

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

apdf = pd.read_csv(annotatedpoints)

# Get all the video_ids 
video_id_list = apdf["video_id"].unique()

# Work on one video_id at a time 
for video_id in video_id_list: 
    partdf = apdf[apdf["video_id"] == video_id]
    
    framenames = [f for f in os.listdir(video_flow_folder) if video_id in f and "mirrored" not in f]
    framenames.sort()

    flowfiles = [os.path.join(video_flow_folder, f) for f in framenames]

    # Retrieve the optical flow of the first annotated frame 
    currentofname = os.path.join(video_flow_folder, partdf["video_frame_id"].iloc[0] + ".npy")
    currentflow = np.load(currentofname)

    # Get the coordinates of the first annotated point and its optical flow vector
    j, i = partdf["x_coord"].iloc[0], partdf["y_coord"].iloc[0]
    apflow = currentflow[i, j]

    # Read original videos for parameters of the writer 
    origvidpath = os.path.join(video_folder, video_id + "_extract.mp4")
    originalvideoreader = cv.VideoCapture(origvidpath)

    # Define output video parameters 
    outputname = "out_sense_" + video_id + ".mp4" 
    print(outputname)
    fourcc = cv.VideoWriter_fourcc(*'mp4v')
    fps = originalvideoreader.get(cv.CAP_PROP_FPS)
    frame_width = currentflow.shape[1]
    frame_height = currentflow.shape[0]
    print(origvidpath, "FPS :", fps)
    originalvideoreader.release()

    # Define output video writer 
    output = cv.VideoWriter(os.path.join(outputfolder, outputname), fourcc, fps, (frame_width, frame_height))
    
    # Compare for each frame in the video the optical flow of the annotated flow to the one of the annotated point 
    currentframenb = partdf["frame_number"].iloc[0]
    currentofname = partdf["video_frame_id"].iloc[0]
    inFrame = True # TURN TO TRUE WHEN FINALIZING THE CODE finalcountdown  


    while currentframenb < len(framenames) and inFrame: 

        # Compare the "annotated" optical flow to all the other optical flow vectors 
        compres = compareFlowsToAnnotatedFlow((i,j), currentflow) 

        # Apply a threshold so we know which part of the image moves like the annotated point 
        seuil = np.quantile(compres, 0.85)
        # print(seuil)
        compres = 255*np.where(compres > seuil, 1, 0)
        compres = np.expand_dims(compres, axis = -1)
        compres = np.concatenate((compres, compres, compres), axis = -1)
        compres = np.uint8(compres)
        compres = annot_viz.visualizePoint(compres, [(i,j)])
        # if currentframenb < 5 : 
        #     print("compres check dims : \n", compres[0,0,:])
        #     cv.imshow("output", compres)
        #     cv.waitKey(0)
        #     print("compres.shape :", compres.shape)
        output.write(compres)

        # Determine the next optical flow vector of comparison
        currentframenb += 1

        # Get the new frame of optical flow 
        newofnamelist = os.path.basename(os.path.splitext(currentofname)[0]).split("_")[:-1]
        newofnamelist.append(str(currentframenb))
        newofname = annot_viz.reconstructFilenameFromList(newofnamelist)
        currentofname = os.path.join(video_flow_folder, newofname + ".npy")
        # Set new currentflow 
        currentflow = np.load(currentofname)

        # Get the new vector of comparison 
        newannotated, inFrameList = annot_viz.calculateNewPosition([(i, j)], currentflow)
        inFrame = inFrameList[0]
        if inFrame: 
            i, j = newannotated[0]
    
    output.release() 

    print(video_id, inFrame)