from __future__ import annotations
import torch
from torch import nn
import torch.nn.functional as F
from FISHClass.ModelZoo._CNNModel_fns import train_fn, validation_fn
from types import MethodType

class ConvBlock(nn.Module):
    
    def __init__(self, in_channels, out_channels):
        super().__init__()
        
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2))
        
    def forward(self, X):
        
        return self.block(X)
       
       
class ClassificationCNN(nn.Module):
    def __init__(self, layers: list=[3, 16, 64, 128, 256], in_shape: list=[128, 128], drop_p=0.5, norm_type="dataset", channels=["red", "green", "blue"], mask=False):
        super().__init__()
        
        self.layers = layers
        self.in_shape = in_shape
        self.drop_p = drop_p
        self.norm_type = norm_type
        self.channels = channels
        self.mask = mask
                
        self.features = torch.nn.Sequential()
        for i in range(len(layers)-1):
            self.features.add_module(f"conv{i}", ConvBlock(layers[i], layers[i+1]))
        
        self.last_nodes = int(layers[-1]*(in_shape[0]/2**(len(layers)-1))**2)
            
        self.fc =  nn.Sequential(
            nn.Linear(self.last_nodes, 1000),
            nn.ReLU(),
            nn.Dropout(self.drop_p),
            nn.Linear(1000, 100),
            nn.ReLU(),
            nn.Dropout(self.drop_p),
            nn.Linear(100, 1)
        )
        
        self.train_fn = MethodType(train_fn, self)
        self.validation_fn = MethodType(validation_fn, self)

    #ignore this, only needed to load pickled model
    def train_fn():
        pass
    
    def validation_fn():
        pass

    def forward(self, X, return_feature_space=False, return_prediction=True):
        
        for mod in self.features:
            X = mod(X)
            fs = X.clone()
            
        if return_feature_space and not return_prediction:
            return fs
            
        X = X.reshape(-1, self.last_nodes)
        
        if return_feature_space and return_prediction:
            return fs, self.fc(X)
        
        if not return_feature_space and return_prediction:
            return self.fc(X)
        
        
