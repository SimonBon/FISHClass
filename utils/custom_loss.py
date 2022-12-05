from itertools import zip_longest
import torch
from torch import nn

class PAL():
    
    def __init__(self,z):
        
        self.z = z
    
    def __call__(self, y_hat, area):
        
        sig_y = torch.sigmoid(y_hat)
        
        return torch.pow(torch.abs(area-sig_y), self.z)
        
#BCE with polynomial area loss
class BCEwithPAL():
    
    def __init__(self, z:int, alpha=1):
        
        self.BCE_loss = nn.BCEWithLogitsLoss()
        self.PAL_loss = PAL(z)
        self.alpha = alpha

    def __call__(self, y_hat, y_target, area):
        
        print("PAL: ", self.PAL_loss(y_hat, area))
        print("BCE: ", self.BCE_loss(y_hat, y_target))
        
        return self.BCE_loss(y_hat, y_target) + self.alpha * self.PAL_loss(y_hat=y_hat, area=area)
        
        
class LAL():
    
    def __init__(self):
        pass
    
    def __call__(self, y_hat, area):
        
        sig_y = torch.sigmoid(y_hat)
        
        return torch.clip(1/(1-(torch.abs(sig_y-area)))*(-torch.log(1-torch.abs(sig_y-area))), 0, 100)
        
#BCE with lgistic area loss
class BCEwithLAL():
    
    def __init__(self, alpha=1):
        
        self.BCE_loss = nn.BCEWithLogitsLoss()
        self.LAL_loss = LAL()
        self.alpha = alpha

    def __call__(self, y_hat, y_target, area, verbose=False):
        
        if verbose:        
            print(torch.sigmoid(y_hat), y_target)
            print("LAL: ", self.LAL_loss(y_hat, area))
            print("BCE: ", self.BCE_loss(y_hat, y_target))
        
        return self.BCE_loss(y_hat, y_target) + self.alpha * self.LAL_loss(y_hat=y_hat, area=area)
        

class LAL2():
    
    def __init__(self, v=1):
        pass
    
    def __call__(self, y_hat, area):
        
        sig_y = torch.sigmoid(y_hat)
        return torch.clip(torch.mean(-torch.log(1-torch.abs(sig_y-area))), 0, 100)
        
#BCE with lgistic area loss
class BCEwithLAL2():
    
    def __init__(self, alpha=1):
        
        self.BCE_loss = nn.BCEWithLogitsLoss()
        self.LAL2_loss = LAL2()
        self.alpha = alpha

    def __call__(self, y_hat, y_target, area, verbose=False):
        
        y_hat = y_hat.squeeze()
        loss = self.BCE_loss(y_hat, y_target) + self.alpha * self.LAL2_loss(y_hat=y_hat, area=area)
        
        if torch.isnan(loss):
            print("LOSS IS NAN") 
        
        if verbose:
            print(torch.sigmoid(y_hat), y_target)
            print("LAL2: ", self.alpha * self.LAL2_loss(y_hat, area))
            print("BCE: ", self.BCE_loss(y_hat, y_target))
            print("COMBINED: ", loss)
            
        return loss