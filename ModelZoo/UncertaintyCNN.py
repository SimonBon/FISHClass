import torch
from torch import nn
import torch.nn.functional as F

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
       
       
class UncertaintyCNN(nn.Module):
    def __init__(self, layers=[3, 16, 64, 128, 256], in_shape=[128, 128], drop_p=0.5, norm_type="dataset"):
        
        self.layers = layers
        self.in_shape = in_shape
        self.drop_p = drop_p
        self.norm_type = norm_type
        
        self.kwargs = {k: v for k, v in self.__dict__.items()}
        super().__init__()
        
        self.features = torch.nn.Sequential()
        for i in range(len(layers)-1):
            self.features.add_module(f"conv{i}", ConvBlock(layers[i], layers[i+1]))
        
        self.last_nodes = int(layers[-1]*(in_shape[0]/2**(len(layers)-1))**2)
        
            
        self.fc =  nn.Sequential(
            nn.Linear(self.last_nodes, 1000),
            nn.ReLU(),
            nn.Dropout(0.25),
            nn.Linear(1000, 100),
            nn.ReLU(),
            nn.Dropout(0.25),
            nn.Linear(100, 1)
        )
        

    def forward(self, X):
        
        for mod in self.features:
            X = mod(X)
            
        X = X.reshape(-1, self.last_nodes)
        return self.fc(X)


    def uncertainty(self):
        
        self.features.eval()
        self.fc.eval()
        
        for m in self.fc.modules():
            if m.__class__.__name__.startswith('Dropout'):
                m.train()
        
        
    def predict_uncertainty(self, X, n=10):
        
        
        Xs = []
        for _ in range(n):
            Xs.append(torch.sigmoid(self.forward(X).squeeze()))
        
        Xs = torch.stack(Xs)
        Xs = Xs.mean(axis=0)
        Xs = torch.stack([Xs, 1-Xs])

        return Xs
    
    
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
        
        
    def train_fn(self, train_loader, loss_fn, optimizer, scheduler, device):
        
        train_loss = 0
        for X, y in train_loader:
            
            ŷ = self.forward(X.to(device))
            loss = loss_fn(ŷ, y.to(device))
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            if not isinstance(scheduler, type(None)):
                scheduler.step()
            
            train_loss += (loss.item()/len(y))
            
        ret = {"train_loss": train_loss}
            
        return ret
    
    def validation_fn(self, validation_loader, loss_fn, device):
            
        self.eval()
        
        val_loss = 0
        accuracy, n = 0, 0
        with torch.no_grad():

            for X, y in validation_loader:
                
                n += len(y)
                ŷ = self.forward(X.to(device))
                val_loss += loss_fn(ŷ, y.to(device)).item()/len(y)
                accuracy += sum((ŷ>0) == y.to(device))

        ret_dict = {"val_loss": val_loss,
                    "accuracy": (accuracy/n).item()*100}
        
        self.train() 
        
        return ret_dict