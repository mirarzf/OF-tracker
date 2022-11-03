import cv2 as cv

### Utility functions 

def reconstructFilenameFromList(name_elements): 
    filename = name_elements[0]
    for e in name_elements[1:]: 
        filename += "_"
        filename += e
    return filename

def visualizePoint(img, coordlist, color=[(0,0,255)]): 
    '''
    Given the i and j coordinates of the pixel of the image img, return an image with img drawn with a red point 
    '''
    # Draw a point at center (i,j)
    for index in range(len(coordlist)): 
        i, j = coordlist[index]
        cv.circle(img, (j,i), 5, (int(color[index][0]), int(color[index][1]), int(color[index][2])), 2)
    return img 

def calculateNewPosition(coordlist, flow): 
    ''' 
    In: 
    List of annotated points coordlist [(i1, j1), (i2, j2), ..., (in, jn)] to calculate new positions in the next frame. 
    Out: 
    newcoordlist: list of pseudo labeled points in the next frame (if they are still in frame). 
    Length of the list is inferior or equal to the input list. 
    inFrame: list of booleans to say if the annotated point is in frame. 
    '''
    newcoordlist, inFrame = [], []
    framewidth = flow.shape[1]
    frameheight = flow.shape[0]
    for coord in coordlist: 
        i, j = coord
        newi, newj = i + flow[i, j, 0], j + flow[i, j, 1]
        if newi < frameheight and newj < framewidth and newi > 0 and newj > 0: 
            newcoordlist.append((int(newi), int(newj)))
            inFrame.append(True)
        else: 
            inFrame.append(False)
    
    return newcoordlist, inFrame
