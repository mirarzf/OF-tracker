import torch 

import torchvision.transforms as T 
import torchvision.transforms.functional as F

class GeometricTransform:
    """
    Geometric transformations to apply for data augmentation 
    """

    def __init__(self):
        """
        Only geometric transformations that are applied are: 
        - Horizontal flipping (horizontal mirroring)
        """
        super().__init__()
        self.transformslist = [
            F.hflip
        ]
        

    def __call__(self, sampledict: dict):
        retdict = {}
        keys = sampledict.keys()
        prob = torch.rand(len(self.transformslist))
        for k in keys: 
            retdict[k] = sampledict[k]
        for t, p in zip(self.transformslist, prob): 
            if p > 0.5: 
                for k in keys: 
                    retdict[k] = t(retdict[k])
        return retdict

class KUTransform:
    """
    Color/brightness transformations to apply to input data for data augmentation 
    """

    def __init__(self):
        super().__init__()
        self.brightnessfactor = 2*torch.rand(1).item()

    def __call__(self, sample):
        """
        Transform the sample when called

        Parameters
        ----------
        sample : PIL.Image or torch.Tensor 
            The image to augment 

        Returns
        -------
         : PIL.Image or torch.Tensor 
            The image after color transformations added 
        """
        return F.adjust_brightness(sample, self.brightnessfactor)