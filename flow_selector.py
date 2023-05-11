from pathlib import Path 
import shutil 
# THIS CODE IS SUPPOSED TO BE PLACED IN 'data'

framesfolder = Path('./GTEA/frames')
toKeepFolder = Path('./GTEA/flowsCorrespondingToFrames')
offolder = Path('./GTEA/of')
folderID = 'S1_Cheese_C1'

filenames = [f.stem for f in framesfolder.listdir() if f.is_file()]
videodirs = [offolder / dir for dir in offolder.iterdir() if (offolder/dir).is_dir()]
print(videodirs) 
for dir in videodirs: 
    print(dir)

    framecount = 20 
    flows = [array for array in dir.iterdir() if array.is_file()]

    vidfilenames = [os.path.basename(f) for f in os.listdir(dir)]
    for filename in vidfilenames: 
        if filename in filenames: 
            cdfilename = os.path.join(dir, filename)
            destfilename = os.path.join(toKeepFolder, filename)
            shutil.copyfile(cdfilename, destfilename)
            print(filename)
