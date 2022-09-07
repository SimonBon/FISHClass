from torch.utils.data import Dataset
import os
import torch
import h5py
import numpy as np
from torchvision.transforms import ToTensor
from FISHClass.utils import custom_transforms as t

class FRCNN_MYCN(Dataset):
    
    def __init__(self, dataset_path: os.PathLike, dataset: str, use_transform=True):
        super().__init__()
        self.dataset_path = dataset_path
        self.dataset = dataset
        self.len = self.get_len()
        self.keys = self.get_keys()
        self.transforms = [t.RandomRotation(), t.RandomFlip()]
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