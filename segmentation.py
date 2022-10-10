from deepcell.utils.plot_utils import make_outline_overlay, create_rgb_image
from deepcell.applications import NuclearSegmentation
import torch
from cellpose.models import Cellpose
import numpy as np

class Segmentation():
    
    #usafe of NuclearSegmentation from DeepCell: https://deepcell.readthedocs.io/en/master/API/deepcell.applications.html#nuclearsegmentation
    def __init__(self, type, *args, **kwargs):
        
        if not type.lower() in ["cellpose", "deepcell"]:
            raise ValueError("Only Cellpose and DeepCell are valid arguments for 'type'")
        
        if type.lower() == "deepcell":
            self.app = NuclearSegmentation(*args, **kwargs)
        
        else:
            device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
            #self.app = Cellpose(gpu=torch.cuda.is_available(), model_type="nuclei", device=device)
            self.app = Cellpose(gpu=torch.cuda.is_available(), model_type="nuclei", device=device)

    # call instance to segment an image
    def __call__(self, im, *args, return_outline=False, MCIm=None, **kwargs):
    

        # if single 2D image is provided increase to 4d with dimensions [1, W, H, 1]
        if im.ndim == 2:
            tmp = np.expand_dims(im, axis=0)
            tmp = np.expand_dims(tmp, axis=-1)
            
        # if mutliple 2d images are given increase to 4d with dimensions [n, W, H, 1]
        elif im.ndim == 3:
            
            if im.shape[-1] == 3: 
                raise ValueError("RGB Images are not supported")
    
            tmp = np.expand_dims(im, axis=-1)
            
        # if already 4D only copy input
        elif im.ndim == 4:
            tmp = np.copy(im)
            
        # call nuclear segmentation from deepcell with args and kwargs
        
        if isinstance(self.app, Cellpose):
                
            masks = self.app.eval(tmp, *args, **kwargs)[0]
            masks = np.expand_dims(masks, axis=0)
            masks = np.expand_dims(masks, axis=-1)
        
        elif isinstance(self.app, NuclearSegmentation):
        
            masks = self.app.predict(tmp, *args, **kwargs)
        
        # for debugging return_outline controlls if an overlay image of cells with outline is retured
        if return_outline:
            # if MCIm is given the rgb image is overlayd with the outlines
            if MCIm:
                if im.ndim > 2:
                    print("No outline can be returned for stacks of images")
                    return im, masks.squeeze(), None
                outline = self.create_outline(tmp, masks, MCIm)
                return im, masks.squeeze(), outline
            else:
                outline = self.create_outline(tmp, masks)
                return im, masks.squeeze(), outline
            
        else:
            return im, masks.squeeze()
        
    
    # create outline image from deepcell.plot_utils
    @staticmethod
    def create_outline(im, mask, MCIm=None):
        
        if MCIm:
            rgb = np.expand_dims(MCIm.RGB, 0)
            outline = make_outline_overlay(rgb, mask)
            
        else:
            rgb = create_rgb_image(im, ["blue"])
            outline = make_outline_overlay(rgb, mask)
        
        return outline.squeeze()
        
         
            
        