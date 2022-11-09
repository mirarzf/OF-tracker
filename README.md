# OF-tracker
Better hand segmentation in egocentric videos thanks to optical flow. 

## Visualization for annotated points and pseudo labels 
The file `annotationvisualization.py` allows a visualization of the annotated points listed in a csv file precised with the argument --dataset. The pth files corresponding to optical flow are stored in the folder given at the head of the files under variable `model_folders`. 
```
python annotationvisualization.py --model raft-things.pth --dataset centerpointstest.csv 
```