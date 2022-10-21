import os 
import pandas as pd 

import numpy as np 
import matplotlib.pyplot as plt 
import cv2 as cv 

video_flow_folder = "C:\\Users\\hvrl\\Documents\\data\\KU\\of" 
video_masks_folder = "C:\\Users\\hvrl\\Documents\\data\\KU\\masks"
annotatedpoints = "C:\\Users\\hvrl\\Documents\\data\\KU\\centerpoints.csv"
video_folder = "C:\\Users\\hvrl\\Documents\\data\\KU\\videos"

outputfolder = ".\\results"

def reconstructFilenameFromList(name_elements): 
    filename = name_elements[0]
    for e in name_elements[1:]: 
        filename += "_"
        filename += e
    return filename

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
    x, y = partdf["x_coord"].iloc[0], partdf["y_coord"].iloc[0]
    currentofname = os.path.join(video_flow_folder, partdf["video_frame_id"].iloc[0] + ".npy")
    currentflow = np.load(currentofname)
    apflow = currentflow[x, y]

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
    print(origvidpath, fps)
    originalvideoreader.release()

    # Define output video writer 
    output = cv.VideoWriter(os.path.join(outputfolder, outputname), fourcc, fps, (frame_width, frame_height))
    
    # Compare for each frame in the video the optical flow of the annotated flow to the one of the annotated point 
    currentframenb = partdf["frame_number"].iloc[0]
    currentofname = partdf["video_frame_id"].iloc[0]
    inFrame = True 
    while currentframenb < len(framenames) and inFrame: 
        # Compare the "annotated" optical flow to all the other optical flow vectors 
        compmat = np.ones(currentflow.shape, dtype = int)*apflow
        compres = 255*np.sum(compmat*currentflow, axis = 2)+255
        compres = compres.astype(np.float32) 
        if currentframenb < 5 : 
            cv.imshow("output", compres)
            cv.waitKey(0)
            print(compres.shape)
            print(compres)
        output.write(cv.cvtColor(compres, cv.COLOR_GRAY2BGR))

        # Determine the next optical flow vector of comparison
        currentframenb += 1

        # Get the new frame of optical flow 
        newofnamelist = os.path.basename(os.path.splitext(currentofname)[0]).split("_")[:-1]
        newofnamelist.append(str(currentframenb))
        newofname = reconstructFilenameFromList(newofnamelist)
        currentofname = os.path.join(video_flow_folder, newofname + ".npy")
        # Set new currentflow 
        currentflow = np.load(currentofname)

        # Get the new of vector of comparison 
        newx = int(x + apflow[0]) 
        newy = int(y + apflow[1]) 
        x = newx 
        y = newy 
        # Set new annotated point optical flow if it is still in frame. Otherwise stop the loop. 
        if x >= 0 and x < currentflow.shape[0] and y >= 0 and y < currentflow.shape[1]: 
            apflow = currentflow[x, y] 
        else: 
            inFrame = False
    
    output.release() 

    print(video_id, inFrame)