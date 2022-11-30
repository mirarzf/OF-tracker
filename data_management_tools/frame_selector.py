import os 
import shutil 

framesfolder = 'D:\\Master Thesis\\data\\KU\\test'
effolder = '.\\extracted_frames'
toKeepFolder = os.path.join(effolder, 'to_keep')
if not os.path.exists(toKeepFolder): 
    os.mkdir(toKeepFolder)
print(toKeepFolder)

filenames = [os.path.basename(f) for f in os.listdir(framesfolder)]
videodirs = [os.path.join(effolder, dir) for dir in os.listdir(effolder) if 'to_keep' not in dir]
print(videodirs)
for dir in videodirs: 
    print(dir)
    vidfilenames = [os.path.basename(f) for f in os.listdir(dir)]
    for filename in vidfilenames: 
        if filename in filenames: 
            cdfilename = os.path.join(dir, filename)
            destfilename = os.path.join(toKeepFolder, filename)
            shutil.copyfile(cdfilename, destfilename)
            print(filename)
