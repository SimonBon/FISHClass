from torch.utils.data import Dataset
import os
import torch
import h5py
import numpy as np
from torchvision.transforms import ToTensor, transforms
from FISHClass.utils import custom_transforms as t

from typing import Callable, List, Tuple
# from CellClass.CNN.data.transforms import NormalizeSample, NormalizeToDataset

import h5py
import numpy as np

class FRCNN_MYCN(Dataset):
    
    def __init__(self, dataset_path: os.PathLike, dataset: str, use_transform=True):
        super().__init__()
        self.dataset_path = dataset_path
        self.dataset = dataset
        self.len = self.get_len()
        self.keys = self.get_keys()
        self.transforms = [t.RandomRotation(), t.RandomBoxFlip()]
        self.h5py_file = h5py.File( self.dataset_path, 'r')
        self.use_transform = use_transform
        self.tt = ToTensor()

        
    def __getitem__(self, idx):
        
            image = self.tt(np.array(self.h5py_file.get(f"{self.dataset}/{self.keys[idx]}/image")))
            boxes = torch.tensor(np.array(self.h5py_file.get(f"{self.dataset}/{self.keys[idx]}/boxes")))
            labels = torch.tensor(np.array(self.h5py_file.get(f"{self.dataset}/{self.keys[idx]}/labels")))
            
            if self.use_transform:
            
                for transform in self.transforms:
                    #s = time()
                    image, boxes = transform(image, boxes)                
                    #print(type(transform), time()-s)
                    
            return  {"image": image, "boxes": boxes, "labels": labels}
      
    
    def __len__(self):
        return self.len
    
    
    def get_keys(self):
        
         with h5py.File( self.dataset_path, 'r') as fin:
            data = fin[self.dataset]
            return list(data.keys())
        
        
    def get_len(self):
        
        with h5py.File(self.dataset_path, "r") as fin:
            
            data = fin[self.dataset]
            return len(data.keys())
    
class LSTMDataset(Dataset):
    
    def __init__(self, data_dict, targets):
        super().__init__()
        
        self.data = data_dict
        self.targets = targets
        
    def __len__(self):
        return len(self.targets)
    
    def __getitem__(self, idx):
        return torch.tensor(self.data[idx]), torch.tensor(self.targets[idx])
    

class BoxesDataset(Dataset):
    
    def __init__(self, dataset_path: os.PathLike, dataset: str, n:int=None):
        super().__init__()
        self.dataset_path = dataset_path
        self.dataset = dataset
        self.h5py_file = h5py.File( self.dataset_path, 'r')
        self.tt = ToTensor()
        self.n = n
        
        self.data = self.get_data()
        
    def __len__(self):
            
        if self.n:
            return self.n
        else:
            return self.data["X"].shape[0]
        
    def get_data(self):
        
        with  h5py.File( self.dataset_path, 'r') as fin:
            
            X = np.array(fin[self.dataset]["X"]).astype(np.float32)
            y = np.array(fin[self.dataset]["y"]).astype(np.float32)
            
        return {"X": torch.tensor(X), "y": torch.tensor(y)}
        
    def __getitem__(self, idx):
            
        return self.data["X"][idx], self.data["y"][idx]


class MYCN(Dataset):
    """Implementation of Dataset for the MYCN dataset.

    Args:
        path: Path to the HDF5 dataset.
        dataset: One of "train", "val" or "test", indicating the subset of images to use.
        transform: a function/transform that takes in a numpy array and returns a transformed version.
    """

    def __init__(
        self,
        path: str,
        dataset: str,
        transform: Callable = None,
        norm_type: str = None,
        n: int = None,
        channels = ["red", "green", "blue"],
        double_return = False
    ):

        if isinstance(norm_type, type(None)) and isinstance(transform, type(None)):
            self.transform = (transforms.Compose([ToTensor()]))
        
        elif isinstance(norm_type, str) and isinstance(transform, type(None)):
            normer = self.define_norm_type(norm_type)
            self.transform = transforms.Compose([ToTensor(), normer])
        
        elif isinstance(norm_type, type(None)) and isinstance(transform, (list, tuple)):
            self.transform = transforms.Compose([ToTensor(), *transform])
            
        elif isinstance(norm_type, str) and isinstance(transform, (list, tuple)):
            normer = self.define_norm_type(norm_type)
            self.transform = transforms.Compose([ToTensor(), *transform, normer])
                
        else:
            raise ValueError(f"Could not resolve {transform} or {norm_type}")

        super().__init__()

        self.dataset = dataset
        self.root = path
        self.n = n
        self.channels = channels
        self.double_return = double_return

    def __len__(self):
        
        with h5py.File(self.root, "r") as dataset:
            n = len(dataset[self.dataset]["y"])
            
        if isinstance(self.n, int):
            if self.n > n:
                return n
            else:
                return self.n
        return n
            

    def __getitem__(self, index):
        
        idxs = []
        for channel in self.channels:
            if channel == "red":
                idxs.append(0)
            if channel == "green":
                idxs.append(1)
            if channel == "blue":
                idxs.append(2)

        with h5py.File(self.root, "r") as dataset:
            img = dataset[self.dataset]["X"][index]
            target = dataset[self.dataset]["y"][index]

        if self.double_return:
            
            normal_image = ToTensor()(img)
            transformed_img = self.transform(img)
            return (normal_image[idxs, ...], transformed_img[idxs, ...] , target.astype(np.float32))

        else:
            
            img = self.transform(img)
            return img[idxs, ...], target.astype(np.float32)

    def define_norm_type(self, norm_type):
        
        if norm_type == "sample":
            return t.NormalizeSample()
            
        elif norm_type == "dataset":
            return t.NormalizeToDataset()

        elif isinstance(norm_type, type(None)):
            return None
        
        else:
            raise ValueError(f"Normalization type {norm_type} is not available!")