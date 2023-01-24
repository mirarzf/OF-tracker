import torch 

from torchvision import transforms 
import torchvision.transforms.functional as F

class KUTransform:
    """
    Define a custom PyTorch transform to implement 
    Data Augmentation specific to the KU dataset 
    """

    def __init__(self):
        """
        Pass custom parameters to the transform in init

        Parameters
        ----------
        """
        super().__init__()
        # transformations we use 
        self.hflip = transforms.transforms.RandomHorizontalFlip()

    def __call__(self, sample):
        """
        Transform the sample when called

        Parameters
        ----------
        sample : PIL.Image
            The image to augment 

        Returns
        -------
        noise_img : PIL.Image
            The image after transformation added
        """
        brightfactor = 2*torch.rand(1).item()
        # print(brightfactor)
        return F.adjust_brightness(self.hflip(sample), 2*torch.rand(1).item())

