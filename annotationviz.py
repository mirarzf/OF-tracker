import os

import numpy as np 
import pandas as pd
import cv2 as cv

from utils import flow_viz

### Utility functions 

def reconstructFilenameFromList(name_elements): 
    filename = name_elements[0]
    for e in name_elements[1:]: 
        filename += "_"
        filename += e
    return filename

def visualizePoint(img, coordlist): 
    '''
    Given the x and y coordinates of the pixel of the image img, return an image with img drawn with a red point 
    '''
    # Draw a point at center (x,y)
    for coord in coordlist: 
        x, y = coord 
        cv.circle(img, (x,y), 5, (0,0,255), 2)
    return img 

def calculateNewPosition(coordlist, flow, framewidth, frameheight): 
    ''' 
    Given a list of annotated points coordlist [(x1, y1), (x2, y2), ..., (xn, yn)], gives their next position if they are still in frame. If not in frame, default back to (-1,-1))
    '''
    newcoordlist = []
    for coord in coordlist: 
        x, y = coord
        newx, newy = x + flow[x, y, 0], y + flow[x, y, 1]
        if newx < frameheight and newy < framewidth and newx > 0 and newy > 0: 
            newcoordlist.append((newx, newy))
        else: 
            newcoordlist.append((-1, -1))
    
    return newcoordlist


### Folders 

video_flow_folder = "C:\\Users\\hvrl\\Documents\\data\\KU\\of" 
video_masks_folder = "C:\\Users\\hvrl\\Documents\\data\\KU\\masks"
video_frames_folder = "C:\\Users\\hvrl\\Documents\\data\\KU\\frames"
annotatedpoints = "C:\\Users\\hvrl\\Documents\\data\\KU\\centerpoints.csv"
video_folder = "C:\\Users\\hvrl\\Documents\\data\\KU\\videos"

outputfolder = ".\\results"

### Main code 

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
    currentframename = os.path.join(video_frames_folder, partdf["video_frame_id"].iloc[0] + ".png")
    currentflow = np.load(currentofname)
    currentframe = cv.imread(currentframename)
    apflow = currentflow[x, y]

    # Read original videos for parameters of the writer 
    origvidpath = os.path.join(video_folder, video_id + "_extract.mp4")
    originalvideoreader = cv.VideoCapture(origvidpath)

    # Define output video parameters 
    outputname = "out_annotated_" + video_id + ".mp4" 
    print(outputname)
    fourcc = cv.VideoWriter_fourcc(*'mp4v')
    fps = originalvideoreader.get(cv.CAP_PROP_FPS)
    frame_width = currentflow.shape[1]
    frame_height = currentflow.shape[0]
    print(origvidpath, fps)
    originalvideoreader.release()

    # Define output video writer 
    output = cv.VideoWriter(os.path.join(outputfolder, outputname), fourcc, fps, (frame_width*2, frame_height))
    
    # Compare for each frame in the video the optical flow of the annotated flow to the one of the annotated point 
    currentframenb = partdf["frame_number"].iloc[0]
    currentofname = partdf["video_frame_id"].iloc[0]
    inFrame = True 
    while currentframenb < len(framenames) and inFrame: 
        # Draw on the image 
        visualizePoint(currentframe, [(x, y)])
        flowimg = flow_viz.flow_to_image(currentflow, convert_to_bgr=True)
        visualizePoint(flowimg, [(x, y)])
        concatenation = cv.hconcat([currentframe, flowimg])
        # cv.imshow("Annotation", flowimg)
        # cv.waitKey(100)
        output.write(concatenation)

        # Determine the next optical flow vector of comparison
        currentframenb += 1

        # Get the new frame of optical flow 
        newofnamelist = os.path.basename(os.path.splitext(currentofname)[0]).split("_")[:-1]
        newofnamelist.append(str(currentframenb))
        newofname = reconstructFilenameFromList(newofnamelist)
        currentofname = os.path.join(video_flow_folder, newofname + ".npy")
        currentframename = os.path.join(video_frames_folder, newofname + ".png")
        # Set new currentflow and new currentframe 
        currentflow = np.load(currentofname)
        currentframe = cv.imread(currentframename)
 
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