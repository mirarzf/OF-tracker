# OF-tracker
Better hand segmentation in egocentric videos thanks to optical flow. 

## Visualization for annotated points and pseudo labels 
The file `annotationvisualization.py` allows a visualization of the annotated points listed in a csv file precised with the argument --dataset. It then creates the pseudo-labelling of the annotated points using optical flow. 
The pth files corresponding to optical flow are stored in the folder given at the head of the files under variable `model_folders`. 
The annotated videos are stored in the folder given at the head of the files under variable `video_folder` but can be changed with the argument --videofolder. 
```(python)
python annotationvisualization.py --model raft-things.pth --dataset centerpointstest.csv 
```

## Attention maps created with optical flow 
The file `flowcomparison.py` creates an attention map using the annotated points listed in a csv file precised with the argument --dataset. It then creates the pseudo-labelling of the annotated points using optical flow. 
The pth files corresponding to optical flow are stored in the folder given at the head of the files under variable `model_folders`. 
The annotated videos are stored in the folder given at the head of the files under variable `video_folder` but can be changed with the argument --videofolder. 
```(python)
python annotationvisualization.py --model raft-things.pth --dataset centerpointstest.csv 
```