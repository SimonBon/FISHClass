from torch import nn
import FISHClass
from FISHClass.utils.data import out2np
import torch
from pathlib import Path
from FISHClass.utils.evaluation import get_top_model, model_from_file

class WeightedFeaturespaceClassifier(nn.Module):
    
    def __init__(self, cnnmodel_path, boxmodel_path, classifiermodel_path, device="cuda", out_channel=32, box_featurespace_size=600, drop_p=0.5, sigmoid=False):
        
        self.cnnmodel_path = cnnmodel_path
        self.boxmodel_path = boxmodel_path
        self.classifiermodel_path = classifiermodel_path
        self.out_channel = out_channel
        self.box_featurespace_size = box_featurespace_size
        self.drop_p = drop_p
        self.sigmoid = sigmoid

        self.kwargs = {k: v for k, v in self.__dict__.items()}

        super().__init__()
        self.device = device
        
        self.box_model, self.cnn_model,  self.classifier_model = self.__define_models(cnnmodel_path, boxmodel_path, classifiermodel_path)
        
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
        if not isinstance(self.classifier_model, FISHClass.ModelZoo.LSTMModel.LSTMClassifier):
            box_fs = torch.flatten(box_fs, start_dim=1).detach()
        else:
            box_fs = box_fs.detach()
            
        basic_pred = self.classifier_model(box_fs).detach()
        
        if self.sigmoid:
            cnn_pred = torch.sigmoid(cnn_pred).detach()
            basic_pred = torch.sigmoid(basic_pred).detach()
            
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
    
    
    def __define_models(self, cnnmodel_path, boxmodel_path, classifiermodel_path):
        
        def get_model(path, device):
            
            if path.is_file():
                model = model_from_file(str(path)).to(device)
            elif path.is_dir():
                model = get_top_model(str(path)).to(device)
                
            return model  
        
        cnn_model = get_model(Path(cnnmodel_path), self.device)
        box_model = get_model(Path(boxmodel_path), self.device)
        classifier_model = get_model(Path(classifiermodel_path), self.device)

        return box_model, cnn_model, classifier_model
        

    def redefine_device(self, device):
        
        self.device = device        
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