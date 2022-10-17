from lib2to3.pytree import convert
import os 

import numpy as np
import cv2 as cv 

from utils import flow_viz

offolder = "C:\\Users\\hvrl\\Documents\\data\\KU\\of"
originalvidfolder = "C:\\Users\\hvrl\\Documents\\data\\KU\\videos"
outputfolder = ".\\results"

video_ids = [
    # '0838_0917_extract', 
    '2108_2112_extract', 
    '5909_5915_extract', 
    # 'green0410_0452_extract'
]

for video_id in video_ids:
    # Read original videos for parameters of the writer 
    origvidpath = os.path.join(originalvidfolder, video_id + ".mp4")
    print(origvidpath)
    originalvideoreader = cv.VideoCapture(origvidpath)

    # Define output video parameters 
    outputname = "out_" + video_id + ".avi" 
    fourcc = cv.VideoWriter_fourcc(*'XVID')
    fps = originalvideoreader.get(cv.CAP_PROP_FPS)
    frame_width = int(originalvideoreader.get(cv.CAP_PROP_FRAME_WIDTH))
    frame_height = int(originalvideoreader.get(cv.CAP_PROP_FRAME_HEIGHT))
    originalvideoreader.release()

    # Define video writer 
    output = cv.VideoWriter(os.path.join(outputfolder, outputname), fourcc, fps, (frame_width, frame_height))

    # List all the optical flow npys 
    listof = [os.path.join(offolder, f) for f in os.listdir(offolder) if video_id in f and "mirrored" not in f] 
    # print(listof)

    # Convert npy optical flow frames to avi video of optical flow 
    for npyname in listof: 
        npy = np.load(npyname)
        flow = np.concatenate((npy, np.zeros((npy.shape[0], npy.shape[1], 1)) ), axis=2)
        # print(flow.shape)
        flowimg = flow_viz.flow_to_image(npy, convert_to_bgr=True)
        # cv.imshow('window', flowimg)
        # cv.waitKey(0) 
        output.write(flowimg)

    output.release()