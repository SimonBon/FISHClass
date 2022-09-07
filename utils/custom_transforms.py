from torch import nn
import numpy as np
from torchvision import transforms
import torch
from torchvision.transforms import InterpolationMode

import matplotlib.pyplot as plt
from CellClass.utils import gridPlot

class RandomFlip(nn.Module):
    
    def __init__(self, ps=(0.5, 0.5)):
        super().__init__()
        self.ps = ps
        self.vert = transforms.RandomVerticalFlip(p=1)
        self.hor = transforms.RandomHorizontalFlip(p=1)
        
    def __call__(self, image, boxes):
        
        boxes = boxes.clone()
        image = image.clone()
        var = np.random.uniform(0,1,2)
        if var[0] <= self.ps[0]:
            image = self.vert(image)
            boxes[:,1] = -boxes[:,1]+image.shape[1]
            boxes[:,3] = -boxes[:,3]+image.shape[1]
            
        if var[1] <= self.ps[1]:
            image = self.hor(image)
            boxes[:,0] = -boxes[:,0]+image.shape[2]
            boxes[:,2] = -boxes[:,2]+image.shape[2]
            
        ret_boxes = []    
        for box in boxes:
            if max(box[0], box[2]) - min(box[0], box[2]) > 0 and max(box[1], box[3]) - min(box[1], box[3]) > 0:
                ret_boxes.append([min(box[0], box[2]), min(box[1], box[3]), max(box[0], box[2]), max(box[1], box[3])])
            

        return image, torch.tensor(ret_boxes)
    
class RandomRotation(nn.Module):
    
    def __init__(self, ps=(0.5, 0.5)):
        super().__init__()
        self.ps = ps
        self.affine = transforms.RandomAffine(
            degrees=(0,360),
            translate=(0,0.1),
            scale=(0.9,1.1),
            interpolation=InterpolationMode.NEAREST)
        
    def __call__(self, image, boxes):
        
        boxes = boxes.clone()
        image = image.clone()
        box_ims = []
        for box in boxes:
            box_im = torch.zeros_like(image)[0]
            box_im[box[1]:box[3], box[0]:box[2]] = 1
            box_ims.append(box_im)
        box_ims = torch.stack(box_ims)
        tmp = torch.cat((image, box_ims))
        image = self.affine(tmp)
        
        boxes = image[3:]
        image = image[:3]
        
        boxes_coords = self.get_box_coords(boxes)

        ret_boxes = []    
        for box in boxes_coords:
            ret_boxes.append([min(box[0], box[2]), min(box[1], box[3]), max(box[0], box[2]), max(box[1], box[3])])
            
        boxes_coords = torch.tensor(ret_boxes)
        
        return image, boxes_coords
    
    def get_box_coords(self, box_ims):
        
        box_ims.bool()
        boxes = []
        for box in box_ims:
            idxs = torch.where(box)
            if len(idxs[0]) == 0:
                continue
            y = idxs[0].min(), idxs[0].max()
            x = idxs[1].min(), idxs[1].max()
            boxes.append([int(x[0]), int(y[0]), int(x[1]), int(y[1])])
            
        return torch.tensor(boxes)
            