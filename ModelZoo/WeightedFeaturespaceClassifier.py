from torch import nn
from FISHClass.utils.data import out2np
import torch
import os
from FISHClass.ModelZoo.FasterRCNNModel import FasterRCNN as frcnn
from FISHClass.ModelZoo.ClassificationCNN import ClassificationCNN as ccnn
from FISHClass.ModelZoo.LSTMModel import LSTMClassifier as lstm
from FISHClass.ModelZoo.BasicModel import BasicClassifier as basic

from FISHClass.ModelZoo._FeaturespaceModel_fns import train_fn, validation_fn
from types import MethodType
from FISHClass.utils.evaluation import get_top_model


class WeightedFeaturespaceClassifier(nn.Module):
    
    def __init__(self, CNNModel, FasterRCNN, ClassifierModel, device="cpu", out_channel=32, box_featurespace_size=600, drop_p=0.5):
        super().__init__()
        
        self.out_channel = out_channel
        self.box_featurespace_size = box_featurespace_size
        self.drop_p = drop_p
        self.device = device
        
        if isinstance(FasterRCNN, frcnn):
            self.box_model = FasterRCNN
        elif isinstance(FasterRCNN, dict):
            self.box_model = FasterRCNN["model"]
        elif isinstance(FasterRCNN, str):
            if os.path.isdir(FasterRCNN):
                self.box_model = torch.load(get_top_model(FasterRCNN))["model"]
            else: 
                self.box_model = torch.load(FasterRCNN)
                if isinstance(FasterRCNN, dict):
                    self.box_model = FasterRCNN["model"]
                
        if isinstance(CNNModel, ccnn):
            self.cnn_model = CNNModel
        elif isinstance(CNNModel, dict):
            self.cnn_model = CNNModel["model"]
        elif isinstance(CNNModel, str):
            if os.path.isdir(CNNModel):
                self.cnn_model = torch.load(get_top_model(CNNModel))["model"]
            else: 
                self.cnn_model = torch.load(CNNModel)
                if isinstance(CNNModel, dict):
                    self.cnn_model = CNNModel["model"]
                
        if isinstance(ClassifierModel, (lstm, basic)):
            self.classifier_model = ClassifierModel
        elif isinstance(ClassifierModel, dict):
            self.classifier_model = ClassifierModel["model"]
        elif isinstance(ClassifierModel, str):
            if os.path.isdir(ClassifierModel):
                self.classifier_model = torch.load(get_top_model(ClassifierModel))["model"]
            else: 
                self.classifier_model = torch.load(ClassifierModel)
                if isinstance(ClassifierModel, dict):
                    self.classifier_model = ClassifierModel["model"]
                       
        self.norm_type = self.cnn_model.norm_type
        self.channels = self.cnn_model.channels
        self.mask = self.cnn_model.mask
        
        last_conv_size = self.cnn_model.features[-1].block[0].out_channels 
        self.conv = nn.Conv2d(in_channels=last_conv_size, out_channels=out_channel, kernel_size=3, padding=1)
        first_fc_size = int(((self.cnn_model.in_shape[0]/2**len(self.cnn_model.features))**2)*out_channel)

        self.fc_cnn = nn.Sequential(
            nn.Linear(first_fc_size, 100),
            nn.ReLU(),
            nn.Dropout(drop_p),
            nn.Linear(100, 100),
            nn.ReLU(),
            nn.Dropout(drop_p),
            nn.Linear(100, 1),
            )
        
        self.fc_classifier = nn.Sequential(
            nn.Linear(box_featurespace_size, 100),
            nn.ReLU(),
            nn.Dropout(drop_p),
            nn.Linear(100, 100),
            nn.ReLU(),
            nn.Dropout(drop_p),
            nn.Linear(100, 1)
        )
        
        self.box_model.to(device)
        self.cnn_model.to(device)


        self.train_fn = MethodType(train_fn, self)
        self.validation_fn = MethodType(validation_fn, self)

    #ignore this, only needed to load pickled model
    def train_fn():
        pass
    
    def validation_fn():
        pass
    
    def forward(self, X, X2, return_details=False, uncertainty=False):
        
        self.box_model.eval()
        self.cnn_model.eval()
        self.classifier_model.eval()
        
        if uncertainty:

            for c in self.children():
                for m in c.modules():
                    if "dropout" in m.__class__.__name__.lower():
                        m.train()
        
        self.cnn_model.requires_grad=False
        self.box_model.requires_grad=False
        self.classifier_model.requires_grad=False
        
        cnn_fs, cnn_pred = self.cnn_model(X2, return_feature_space=True, return_prediction=True)
        cnn_fs = cnn_fs.detach()
        cnn_pred = cnn_pred.detach().squeeze()
        box_fs = self.box_model(X)
        
        box_fs = out2np(box_fs, device=self.device)
        if not isinstance(self.classifier_model, lstm):
            box_fs = torch.flatten(box_fs, start_dim=1).detach()
        else:
            box_fs = box_fs.detach()
            
        basic_pred = self.classifier_model(box_fs).detach()
            
        cnn_fs = self.conv(cnn_fs)
        cnn_fs = torch.flatten(cnn_fs, start_dim=1)
        
        box_fs = torch.flatten(box_fs, start_dim=1).detach()
        weight_cnn = self.fc_cnn(cnn_fs).squeeze()
        weight_box = self.fc_classifier(box_fs).squeeze()

        weighted_cnn = weight_cnn * cnn_pred
        weighted_box = weight_box * basic_pred
        
        final_pred = weighted_cnn + weighted_box
        
        if return_details:
            return {"final_pred": final_pred,
                    "cnn_pred": cnn_pred, 
                    "cnn_weight": weight_cnn,
                    "basic_pred": basic_pred,
                    "basic_weight": weight_box}
        
        return final_pred  

    def redefine_device(self, device):
        
        self.device = device     
        self.to(device)   
        self.box_model.to(device)
        self.cnn_model.to(device)
        self.classifier_model.to(device)
        
        
    def predict_uncertainty(self, X, X2, n=10):
            
        Xs = []
        for _ in range(n):
            Xs.append(torch.sigmoid(self.forward(X, X2, uncertainty=True).squeeze()))
        
        Xs = torch.stack(Xs)
        Xs = Xs.mean(axis=0).cpu().detach().numpy()

        return Xs