import argparse
import os
import cv2
import glob
import numpy as np


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--video_input', required=True, help='give the video file input')
    args = parser.parse_args()
    video_input_name = os.path.basename(args.video_input).split(".")[0]
    print(video_input_name)
    
    cap = cv2.VideoCapture(args.video_input)
    compteur = 0 
    ret, frame = cap.read() 
    save_folder = os.path.join("extracted_frames", video_input_name)
    if not os.path.exists(save_folder): 
        os.mkdir(save_folder)
    while ret: 
        compteur += 1 
        save_path = os.path.join(save_folder, video_input_name + "_" + str(compteur) + ".png")
        cv2.imwrite(save_path, frame)
        ret, frame = cap.read() 
    cap.release() 

 