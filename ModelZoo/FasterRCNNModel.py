import torch
from torch import nn
from torchvision.models.detection import fasterrcnn_resnet50_fpn 
from torchvision.models.detection import FasterRCNN as frcnn
import torch 

class FasterRCNN(frcnn):
    
    def __init__(self, state_dict=None, weights=None, channels=["red", "green", "blue"], mask=False, norm_type=None):
        
        self.channels = channels
        self.mask = mask
        self.norm_type = norm_type
        
        if weights is not None:
            print("pretrained")
            model = fasterrcnn_resnet50_fpn(pretrained=True, weights=weights)
            
        else:
            print("not_pretrained")
            model = fasterrcnn_resnet50_fpn(weights=None)

        super().__init__(backbone=model.backbone, num_classes=91)
            
        if not isinstance(state_dict, type(None)):
            print("load_state_dict")
            self.load_state_dict(state_dict)


    def train_fn(self, train_loader, loss_fn, optimizer, scheduler, device):
        
        train_loss = 0
        for X, y in train_loader:
            
            X = [x.to(device) for x in X]
            y = [{k: v.to(device) for k,v in sub_y.items()} for sub_y in y]
            
            ŷ = self.forward(X, y)
            loss = loss_fn(val for val in ŷ.values())
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            if not isinstance(scheduler, type(None)):
                scheduler.step()
            
            train_loss += loss.item()
            del ŷ, X, y, loss
            
        train_loss = train_loss/len(train_loader)
        ret = {"train_loss": train_loss}
            
        return ret


    def validation_fn(self, validation_loader, loss_fn, device):
        
        val_loss = 0
        with torch.no_grad():

            for X, y in validation_loader:
                
                X = [im.to(device) for im in X]
                y = [{k: v.to(device) for k,v in sub_y.items()} for sub_y in y]

                out = self.forward(X, y)
                
                val_loss += loss_fn(loss for loss in out.values()).item()
                del out, X, y

        val_loss = val_loss/len(validation_loader)
        ret_dict = {"val_loss": val_loss}
        return ret_dict

def train_fn(self, train_loader, loss_fn, optimizer, scheduler, device):
    
    train_loss = 0
    for X, y in train_loader:
        
        X = [x.to(device) for x in X]
        y = [{k: v.to(device) for k,v in sub_y.items()} for sub_y in y]
        
        ŷ = self.forward(X, y)
        loss = loss_fn(val for val in ŷ.values())
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if not isinstance(scheduler, type(None)):
            scheduler.step()
        
        train_loss += loss.item()
        del ŷ, X, y, loss
        
    train_loss = train_loss/len(train_loader)
    ret = {"train_loss": train_loss}
        
    return ret


def validation_fn(self, validation_loader, loss_fn, device):
    
    val_loss = 0
    with torch.no_grad():

        for X, y in validation_loader:
            
            X = [im.to(device) for im in X]
            y = [{k: v.to(device) for k,v in sub_y.items()} for sub_y in y]

            out = self.forward(X, y)
            
            val_loss += loss_fn(loss for loss in out.values()).item()
            del out, X, y

    val_loss = val_loss/len(validation_loader)
    ret_dict = {"val_loss": val_loss}
    return ret_dict