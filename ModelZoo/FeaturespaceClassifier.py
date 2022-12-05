from sys import int_info
from torch import nn
import torch

from FISHClass.utils.data import out2np
from FISHClass.ModelZoo.FasterRCNNModel import FasterRCNN as frcnn
from FISHClass.ModelZoo.ClassificationCNN import ClassificationCNN as ccnn
from FISHClass.ModelZoo._FeaturespaceModel_fns import train_fn, validation_fn

from types import MethodType

class FeaturespaceClassifier(nn.Module):
    
    def __init__(self, CNNModel, FasterRCNN, device="cpu", out_channel=32, box_featurespace_size=600, drop_p=0.5, custom_loss=False, train_cnn=False):
        
        super().__init__()
        
        self.device = device
        self.out_channel = out_channel
        self.box_featurespace_size = box_featurespace_size
        self.drop_p = drop_p
        self.custom_loss = custom_loss
        self.train_cnn = train_cnn
        
        if isinstance(FasterRCNN, frcnn):
            self.box_model = FasterRCNN
        elif isinstance(FasterRCNN, dict):
            self.box_model = FasterRCNN["model"]
        elif isinstance(FasterRCNN, str):
            self.box_model = torch.load(FasterRCNN)
            if isinstance(self.box_model, dict):
                self.box_model = self.box_model["model"]
                
        if isinstance(CNNModel, ccnn):
            self.cnn_model = CNNModel
        elif isinstance(CNNModel, dict):
            self.cnn_model = CNNModel["model"]
        elif isinstance(CNNModel, str):
            self.cnn_model = torch.load(CNNModel)
            if isinstance(self.cnn_model, dict):
                self.cnn_model = self.cnn_model["model"]
        
        self.norm_type = self.cnn_model.norm_type
        self.channels = self.cnn_model.channels
        self.mask = self.cnn_model.mask

        self.last_conv_size = self.cnn_model.features[-1].block[0].out_channels
        
        self.conv = nn.Sequential(nn.Conv2d(in_channels=self.last_conv_size, out_channels=out_channel, kernel_size=3, padding=1))
        
        self.first_fc_size = int(((self.cnn_model.in_shape[0]/2**(len(self.cnn_model.features)))**2)*out_channel) + box_featurespace_size

        print(self.first_fc_size)

        self.fc = nn.Sequential(
            nn.Linear(self.first_fc_size, 1000),
            nn.ReLU(),
            nn.Dropout(self.drop_p),
            nn.Linear(1000, 100),
            nn.ReLU(),
            nn.Dropout(self.drop_p),
            nn.Linear(100, 1)
            )
        
        self.train_fn = MethodType(train_fn, self)
        self.validation_fn = MethodType(validation_fn, self)

        self.redefine_device(device)
    
    def train_fn():
        pass
    
    def validation_fn():
        pass
    
    
    def forward(self, X, X2, verbose:bool = False, verbose_threshold: float = 0.5, return_box_fs: bool = False):
        
        self.box_model.eval()
        self.box_model.requires_grad=False
        
        cnn_out = self.cnn_model(X2, return_feature_space=True, return_prediction=False)
        if not self.train_cnn:
            cnn_out = cnn_out.detach()
        box_out = self.box_model(X)
            
        box_feature_space = out2np(box_out, device=self.device)
        
        if verbose:
            for ii, im in enumerate(box_feature_space):
                red_spots = (torch.logical_and(im[:, 4] == 1, im[:,5]> verbose_threshold)).sum().detach().cpu().numpy()
                green_spots = (torch.logical_and(im[:, 4] == 2, im[:,5]> verbose_threshold)).sum().detach().cpu().numpy()
                clumps = (torch.logical_and(im[:, 4] == 3, im[:,5]> verbose_threshold)).sum().detach().cpu().numpy()
                
                print(f"{ii}: REDS: {red_spots}, GREENS: {green_spots}, CLUMPS: {clumps}")
            
        box_feature_space = torch.flatten(box_feature_space, start_dim=1)
        box_feature_space = box_feature_space.detach()

        cnn_fs = self.conv(cnn_out)
        cnn_fs = torch.flatten(cnn_fs, start_dim=1)
        
        inp = torch.cat((cnn_fs, box_feature_space), axis=1)
        
        out = self.fc(inp)
        
        if return_box_fs:
            return out, box_feature_space
        
        return out 
    
    
    def redefine_device(self, device):
            
        self.device = device    
        self.to(device)    
        self.box_model.to(device)
        self.cnn_model.to(device)
