from torch import nn
import numpy as np
from torchvision import transforms
import torch
from torchvision.transforms import InterpolationMode

class RandomBoxFlip(nn.Module):
    
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
            scale=(0.8,1.2),
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
    
    


class NormalizeSample(torch.nn.Module):
    """Normalize a tensor image w.r.t. to its mean and standard deviation.

    Args:
        inplace: Whether to make this operation in-place.
    """

    def __init__(self):
        super().__init__()

    def __call__(self, tensor: torch.Tensor) -> torch.Tensor:
        
        for i in range(tensor.shape[0]):
            
            mean = tensor[i][tensor[i]!=0].mean()
            std = tensor[i][tensor[i]!=0].std()
            
            if std == 0:
                tensor[i][tensor[i]!=0] = (tensor[i][tensor[i]!=0]-mean)
            
            tensor[i][tensor[i]!=0] = (tensor[i][tensor[i]!=0]-mean)/std
        
        return tensor
    
    
class RandomAffine(torch.nn.Module):
        
    def __init__(self):
        super().__init__()
        self.transformer = transforms.RandomAffine(degrees=(0,360), translate=(0, 0.2), scale=(0.8, 1.2), fill=0)

    def __call__(self, tensor):
        
        return self.transformer(tensor)
  
  
class RandomFlip(torch.nn.Module):
    
    def __init__(self):
        super().__init__()
        self.vert = transforms.RandomVerticalFlip()
        self.hor = transforms.RandomHorizontalFlip()
    
    def __call__(self, tensor):
        
        tensor = self.vert(tensor)
        tensor = self.hor(tensor)
        
        return tensor
    
    
class RandomNoise():
    
    def __init__(self, mean=0, std=1):
        
        self.mean = mean
        self.std = std
    
    def __call__(self, img):
        
        random_weight = 0.1*torch.rand(1)
        noise = random_weight * torch.normal(self.mean, self.std, img.shape)

        noise[img==0] = 0

        return img + noise
    
class NormalizeToDataset(torch.nn.Module):
    """Normalize a tensor image w.r.t. to mean and standard deviation of the dataset."""
    
    def __init__(self):
        super().__init__()
        #self.means = torch.Tensor([ 0.20248361, 0.29113407, 0.55524067]) #bestimmt von h5/trainset_more_celllines_small.h5 ["train"] dataset
        #self.stds = torch.Tensor([0.09509387, 0.16945897, 0.16422895])
        self.means = torch.Tensor([ 0.14323247, 0.25705834, 0.49026522]) #bestimmt von h5/trainset_small.h5 ["train"] dataset
        self.stds = torch.Tensor([0.04890371, 0.09778837, 0.09606459])

    def __call__(self, tensor: torch.Tensor) -> torch.Tensor:
        
        for i in range(3):
            
            tensor[i][tensor[2]!=0] = (tensor[i][tensor[2]!=0] - self.means[i])/self.stds[i]
        
        return tensor
    
class ReverseNormalize(torch.nn.Module):
    """Normalize a tensor image w.r.t. to mean and standard deviation of the dataset."""
    
    def __init__(self):
        super().__init__()

        #self.means = torch.Tensor([ 0.20248361, 0.29113407, 0.55524067]) #bestimmt von h5/trainset_more_celllines_small.h5 ["train"] dataset
        #self.stds = torch.Tensor([0.09509387, 0.16945897, 0.16422895])
        self.means = torch.Tensor([ 0.14323247, 0.25705834, 0.49026522]) #bestimmt von h5/trainset_small.h5 ["train"] dataset
        self.stds = torch.Tensor([0.04890371, 0.09778837, 0.09606459])

    def __call__(self, tensor: torch.Tensor) -> torch.Tensor:
        # https://stackoverflow.com/questions/32320090/how-to-denormalise-de-standardise-neural-net-predictions-after-normalising-inp][1]
        
        if tensor.ndim == 4:
            
            for im in tensor:
                for i in range(3):
                    im[i][im[i]!=0] = (im[i][im[i]!=0]*self.stds[i]) + self.means[i]
            
        elif tensor.ndim == 3:
    
            for i in range(3):
                tensor[i][tensor[i]!=0] = (tensor[i][tensor[i]!=0]*self.stds[i]) + self.means[i]
        
        return tensor
    
    
class RandomIntensity(torch.nn.Module):
    
    def __init__(self, channels=["red", "green", "blue"]):
        super().__init__()
        
        self.channels = channels
        self._channels_dict = {"red": 0, "green": 1, "blue": 2}
        self.lower_scale_limit = -0.5
        self.upper_scale_limit = 0.5
        
    def __call__(self, tensor: torch.Tensor):
        
        scale_val = np.random.uniform(
            self.lower_scale_limit, 
            self.upper_scale_limit
        )
        
        for key, idx in self._channels_dict.items():
            
            if key in self.channels:
                
                tensor[idx] = tensor[idx]*(1+scale_val)
                torch.clip(tensor, 0, 1, out=tensor)
                
        return tensor