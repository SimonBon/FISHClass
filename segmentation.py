from deepcell.utils.plot_utils import create_rgb_image
from deepcell.applications import NuclearSegmentation
import torch
from cellpose.models import Cellpose
import numpy as np
from skimage.segmentation import find_boundaries
import cv2

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
            print(kwargs)
            self.app = Cellpose(*args, **kwargs, device=device, gpu=torch.cuda.is_available())

    # call instance to segment an image
    def __call__(self, im, *args, return_outline=False, MCIm=None, dilate_iteration=1, **kwargs):
    

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
                outline = self.create_outline(tmp, masks, MCIm, dilate_iteration=dilate_iteration)
                return im, masks.squeeze(), outline
            else:
                outline = self.create_outline(tmp, masks, dilate_iteration=dilate_iteration)
                return im, masks.squeeze(), outline
            
        else:
            return im, masks.squeeze()
        
    
    # create outline image from deepcell.plot_utils
    @staticmethod
    def create_outline(im, mask, MCIm=None, dilate_iteration=1):
        
        if MCIm:
            rgb = np.expand_dims(MCIm.RGB, 0)
            outline = make_outline_overlay(rgb, mask, dilate_iteration)
            
        else:
            rgb = create_rgb_image(im, ["blue"])
            outline = make_outline_overlay(rgb, mask, dilate_iteration)
        
        return outline.squeeze()
        
         
    
def make_outline_overlay(rgb_data, predictions, dilate_iteration=1):
    """Overlay a segmentation mask with image data for easy visualization

    Args:
        rgb_data: 3 channel array of images, output of ``create_rgb_data``
        predictions: segmentation predictions to be visualized

    Returns:
        numpy.array: overlay image of input data and predictions

    Raises:
        ValueError: If predictions are not 4D
        ValueError: If there is not matching RGB data for each prediction
    """
    
    if len(predictions.shape) != 4:
        raise ValueError('Predictions must be 4D, got {}'.format(predictions.shape))

    if predictions.shape[0] > rgb_data.shape[0]:
        raise ValueError('Must supply an rgb image for each prediction')

    boundaries = np.zeros_like(rgb_data)
    overlay_data = np.copy(rgb_data)

    for img in range(predictions.shape[0]):
        boundary = find_boundaries(predictions[img, ..., 0], connectivity=1, mode='inner')
        boundaries[img, boundary > 0, :] = 1
        kernel = np.ones((3, 3), np.uint8)
        boundaries = boundaries[0, ..., 0]
        boundaries = cv2.dilate((boundaries*255).astype(np.uint8), kernel, iterations=dilate_iteration)
        
    overlay_data[0, boundaries > 0, :] = 1

    return overlay_data