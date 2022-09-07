from torchvision.models import resnet50
from torch import nn
import torch

class AdaptedResNet50(nn.Module):
    
    def __init__(self, weights=None, drop_p=0.5):
        super().__init__()
        
        self.weights = weights
        self.drop_p = drop_p
        
        new_fc = nn.Sequential(
            nn.Linear(2048, 1000),
            nn.ReLU(),
            nn.Dropout(self.drop_p),
            nn.Linear(1000,100),
            nn.ReLU(),
            nn.Dropout(self.drop_p),
            nn.Linear(100, 1)
        )
        
        self.base_model = resnet50(weights=self.weights)
        self.remove_grad()
        self.base_model.add_module("fc", new_fc)
        
        
    def forward(self, X):
        
        X = self.base_model(X)
        return X
        

    def remove_grad(self):
        
        for p in self.base_model.parameters():
            p.requires_grad = False
        