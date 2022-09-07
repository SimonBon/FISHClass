from torch import nn
import torch
from pathlib import Path

import FISHClass
from FISHClass.utils.data import out2np
from FISHClass.utils.evaluation import get_top_model, model_from_file

class FeaturespaceClassifier(nn.Module):
    
    def __init__(self, cnnmodel_path, boxmodel_path, device="cuda", out_channel=32, box_featurespace_size=600):
        
        self.cnnmodel_path = cnnmodel_path
        self.boxmodel_path = boxmodel_path
        self.device = device
        self.out_channel = out_channel
        self.box_featurespace_size = box_featurespace_size
        
        self.kwargs = {k: v for k, v in self.__dict__.items()}
        
        super().__init__()
        
        self.box_model, self.cnn_model = self.__define_models(cnnmodel_path, boxmodel_path)
        
        last_conv_size = self.cnn_model.features[-1].block[0].out_channels 
        self.conv = nn.Conv2d(in_channels=last_conv_size, out_channels=out_channel, kernel_size=3, padding=1)
        first_fc_size = int(((self.cnn_model.in_shape[0]/2**len(self.cnn_model.features))**2)*out_channel) + box_featurespace_size

        self.fc = nn.Sequential(
            nn.Linear(first_fc_size, 100),
            nn.ReLU(),
            nn.Linear(100, 100),
            nn.ReLU(),
            nn.Linear(100, 1)
            )
        
        
    def forward(self, X, X2):
        
        self.cnn_model.eval()
        self.box_model.eval()
        
        self.cnn_model.requires_grad=False
        self.box_model.requires_grad=False
        
        cnn_out = self.cnn_model(X2, return_feature_space=True, return_prediction=False)
        cnn_out = cnn_out.detach()
        box_out = self.box_model(X)
        
        box_feature_space = out2np(box_out, device="cuda")
        box_feature_space = torch.flatten(box_feature_space, start_dim=1)
        box_feature_space = box_feature_space.detach()

        cnn_fs = self.conv(cnn_out)
        cnn_fs = torch.flatten(cnn_fs, start_dim=1)
        
        inp = torch.cat((cnn_fs, box_feature_space), axis=1)
        
        out = self.fc(inp)
        
        return out    

    def __define_models(self, cnnmodel_path, boxmodel_path):
        
        def get_model(path, device):
            
            if path.is_file():
                model = model_from_file(str(path)).to(device)
            elif path.is_dir():
                model = get_top_model(str(path)).to(device)
                
            return model  
        
        cnn_model = get_model(Path(cnnmodel_path), self.device)
        box_model = get_model(Path(boxmodel_path), self.device)
            
        return box_model, cnn_model