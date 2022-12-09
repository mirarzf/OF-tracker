import numpy as np
from copy import deepcopy

def normalizeMat(mat): 
    '''
    In: 
    mat: 2D array. 
    
    Out: 
    array with values between 0 and 1 
    '''

    mini = mat.min()
    maxi = mat.max() 
    actualrange = maxi-mini
    
    return (mat-mini*np.ones(mat.shape))/actualrange

def compareFlowsToAnnotatedFlow(ap, flow): 
    '''
    In: 
    aplist: (x,y,type) with x, y coordinates of the annotated point. We compare the optical flow 
    in the complete image to the optical flow at this point. type describes the type of the annotated point: background or hand. 
    flow: numpy array of shape (height, width, 2) corresponding to the optical flow 

    Out: 
    compdot: numpy array of shape (height, width) with values between 0 and 1 
    '''
    apcoord = ap[:2]

    norms = np.sqrt(flow[:,:,0]**2 + flow[:,:,1]**2)
    normapflow = norms[apcoord[0], apcoord[1]]

    apunitmat = flow[apcoord[0], apcoord[1],:]*np.ones(flow.shape)

    compdot = np.sum(apunitmat*flow, axis=2)/norms
    compdot /= normapflow
    # print(compdot.shape)
    # print("compdot result check : ", compdot[apcoord[0], apcoord[1]])

    if ap[2] == 0: # Type of the annotated point is background 
        # return np.ones(compdot.shape)-compdot
        compdot *= -1

    # # If the calculations are correct, values should be between -1 and 1. 
    # # But because of the approximations, the maximum and minimum values found in the 
    # # comparison matrix can be higher. 
    # mincompdot = compdot.min()
    # if mincompdot > -1: 
    #     mincompdot = -1 
    # maxcompdot = compdot.max()
    # if maxcompdot < 1: 
    #     maxcompdot = 1
    # actualrange = maxcompdot - mincompdot 
    
    # compdot = compdot - mincompdot*np.ones(compdot.shape)
    # compdot /= actualrange
    # print(mincompdot, maxcompdot, actualrange)
    # print("compdot result check : ", compdot[apcoord[0], apcoord[1]])

    return normalizeMat(compdot) 




def compareFlowsToMultipleAnnotatedFlows(aplist, flow): 
    '''
    In: 
    aplist: list of (x,y,type) (pseudo-)annotated points to track. 
    flow: vector field repsenting the optical flow. 

    Out: 
    compdotsum: numpy array of shape (height, width) with values between 0 and 1, 
    result of the weighted sum of the comparison maps of the optical flow field with each annotated point flow vector. 
    '''

    compdotreslist = []
    for ap in aplist: 
        compdotreslist.append(compareFlowsToAnnotatedFlow(ap, flow))
    compdotsum = np.sum(np.array(compdotreslist), axis=0)/len(aplist)

    # Normalization of the resulting average map to spread results between 0 and 1 
    mini = compdotsum.min()
    maxi = compdotsum.max()
    actualrange = maxi - mini 
    
    compdotsum = compdotsum - mini*np.ones(compdotsum.shape)
    compdotsum /= actualrange

    return compdotsum