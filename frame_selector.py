import os 
import shutil 

framesfolder = 'D:\\Master Thesis\\data\\KU\\test'
toKeepFolder = './extracted_frames/to_keep/'
if not os.path.exists(toKeepFolder): 
    os.mkdir(toKeepFolder)

filenames = [os.path.basename(f) for f in os.listdir(framesfolder)]
for filename in filenames: 
    cdfilename = os.path.join('/extracted_frames', filename)
    print(shutil.move(cdfilename, toKeepFolder), filename)
