import cv2 as cv
import torch

### Utility functions 

### Main code 
DEVICE = 'cuda'

def load_image(img, destsize): 
    img = cv.resize(img, destsize)
    imgtensor = torch.from_numpy(img).permute(2, 0, 1).float()
    return imgtensor[None].to(DEVICE)

def reconstructFilenameFromList(name_elements): 
    filename = name_elements[0]
    for e in name_elements[1:]: 
        filename += "_"
        filename += e
    return filename

def unscaledCoordlist(coordlist, scale=1): 
    newcoordlist = [(e[0]/scale, e[1]/scale) for e in coordlist]
    return newcoordlist

def visualizePoint(img, coordlist, color=[(0,0,255)], scale = 1): 
    '''
    Given the i and j coordinates of the pixel of the image img, return an image with img drawn with a red point 
    '''
    # Draw a point at center (i,j)
    for index, coord in enumerate(coordlist): 
        i, j = coord
        cv.circle(img, (int(j*scale),int(i*scale)), 5, (int(color[index][0]), int(color[index][1]), int(color[index][2])), 2)
    return img 

def calculateNewPosition(aplist, flow): 
    ''' 
    In: 
    aplist: List of annotated points coordinates and their type [(i1, j1, type1), (i2, j2, type2), ..., (in, jn, typen)] 
    to calculate new positions in the next frame. 
    flow: vector field of the movements of each pixel 
    Out: 
    newcoordlist: list of pseudo labeled points in the next frame (if they are still in frame). 
    Length of the list is inferior or equal to the input list. 
    inFrame: list of booleans to say if the annotated point is in frame. 
    '''
    newcoordlist, inFrame = [], []
    framewidth = flow.shape[1]
    frameheight = flow.shape[0]
    for ap in aplist: 
        i, j, type = ap
        newi, newj = i + flow[i, j, 0], j + flow[i, j, 1]
        if newi < frameheight and newj < framewidth and newi > 0 and newj > 0: 
            newcoordlist.append((int(newi), int(newj), type))
            inFrame.append(True)
        else: 
            inFrame.append(False)
    
    return newcoordlist, inFrame
