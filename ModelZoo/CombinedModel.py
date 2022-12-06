import torch
import os
from torch import nn
from FISHClass.utils.data import out2np

from FISHClass.ModelZoo.FasterRCNNModel import FasterRCNN as frcnn
from FISHClass.ModelZoo.LSTMModel import LSTMClassifier as lstm
from FISHClass.ModelZoo.BasicModel import BasicClassifier as basic
from FISHClass.utils.evaluation import get_top_model 
                
class CombinedModel(nn.Module):
    
    def __init__(self, FasterRCNN, ClassifierModel, device="cpu"):
        super().__init__()
        
        if isinstance(FasterRCNN, frcnn):
            self.box_model = FasterRCNN
        elif isinstance(FasterRCNN, dict):
            self.box_model = FasterRCNN["model"]
        elif isinstance(FasterRCNN, str):
            if os.path.isdir(FasterRCNN):
                self.box_model = torch.load(get_top_model(FasterRCNN))["model"]
            else:
                self.box_model = torch.load(FasterRCNN)["model"]

                
        if isinstance(ClassifierModel, (lstm, basic)):
            self.classifier_model = ClassifierModel
        elif isinstance(ClassifierModel, dict):
            self.classifier_model = ClassifierModel["model"]
        elif isinstance(ClassifierModel, str):
            if os.path.isdir(ClassifierModel):
                self.classifier_model = torch.load(get_top_model(ClassifierModel))["model"]
            else:
                self.classifier_model = torch.load(ClassifierModel)["model"]
            
        self.norm_type = self.box_model.norm_type
        self.mask = self.box_model.mask
        self.channels = self.box_model.channels
        
        self.classifier_model.to(device)  
        self.box_model.to(device)
             
    def forward(self, X): 
            
        self.box_model.eval()
        self.classifier_model.eval()
        
        self.box_model.requires_grad=False
        self.classifier_model.requires_grad = False
        
        box_fs = self.box_model(X)
        box_fs = out2np(box_fs, device=self.device)
            
        pred = self.classifier_model(box_fs)
        
        return pred.squeeze()
        
    def redefine_device(self, device):
        
        self.device = device     
        self.to(device)   
        self.box_model.to(device)
        self.classifier_model.to(device)