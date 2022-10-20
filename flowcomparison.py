import os 
import pandas as pd 

import numpy as np 

video_flow_folder = "c:\\users\\hvrl\\Documents\\data\\KU\\of" 
video_frames_folder = "C:\\Users\\hvrl\\Documents\\data\\KU\\frames"
video_masks_folder = "C:\\Users\\hvrl\\Documents\\data\\KU\\masks"
annotatedpoints = "C:\\Users\\hvrl\\Documents\\data\\KU\\centerpoints.csv"

def reconstructFilenameFromList(name_elements): 
    filename = name_elements[0]
    for e in name_elements[1:]: 
        filename += "_"
        filename += e
    return filename

apdf = pd.read_csv(annotatedpoints)

print(apdf)

# Get all the video_ids 
video_id_list = apdf["video_id"].unique()

# Work on one video_id at a time 
for video_id in video_id_list: 
    partdf = apdf[apdf["video_id"] == video_id]
    print(partdf)
    framenames = [f for f in os.listdir(video_frames_folder) if video_id in f]
    framenames.sort()

    print(framenames[:5])
    framefiles = [os.path.join(video_frames_folder, f) for f in framenames]
    flowfiles = [os.path.join(video_flow_folder, f) for f in framenames]

    # Retrieve the optical flow of the first annotated frame 
    print(partdf["video_frame_id"].iloc[0] + ".npy")
    x, y = partdf["x_coord"].iloc[0], partdf["y_coord"].iloc[0]
    currentofname = os.path.join(video_flow_folder, partdf["video_frame_id"].iloc[0] + ".npy")
    apflow = np.load(currentofname)[x, y]
    print(apflow)
    
    # Compare for each frame in the video the optical flow of the annotated flow to the one of the annotated point 
    currentframenb = partdf["frame_number"].iloc[0]
    currentofname = partdf["video_frame_id"].iloc[0]
    while currentframenb < len(framenames) + 1: 
        # Compare the "annotated" optical flow to all the other optical flow vectors 

        # Determine the next optical flow vector of comparison
        currentframenb += 1

        # Get the new frame of optical flow 
        newofnamelist = os.path.basename(os.path.splitext(currentofname)[0]).split("_")[:-1]
        newofnamelist.append(str(currentframenb))
        newofname = reconstructFilenameFromList(newofnamelist)
        currentofname = os.path.join(video_flow_folder, newofname + ".npy")
        print(currentofname)

        # Get the new of vector of comparison 
        newx = int(x + apflow[0]) 
        newy = int(y + apflow[1]) 
        x = newx 
        y = newy 

    print(currentframenb)