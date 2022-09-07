import torch
from torch import nn
from FISHClass.utils.evaluation import get_top_model, model_from_file
from pathlib import Path


class CombinedModel(nn.Module):
    
    def __init__(self, fasterrcnn_path, basicmodel_path=None, lstmmodel_path=None):
        super().__init__()
        
        self.classification_model, self.box_model = self.__define_models(fasterrcnn_path, basicmodel_path, lstmmodel_path)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
        
    def forward(self, images): 
        
        with torch.no_grad():
            
            X = [im.to(self.device) for im in images]
            self.box_model.to(self.device)
            self.classification_model.to(self.device)
            
            self.box_model.eval()
            boxes = self.box_model(X)
            
            inters = []
            for box in boxes:
            
                pad_sz = 100-len(box["labels"])
                
                inter = torch.cat((box["boxes"], box["labels"].unsqueeze(1), box["scores"].unsqueeze(1)), axis=1)
                inter = torch.cat((inter, torch.zeros(pad_sz, 6).to(self.device)), axis=0)
                inters.append(inter)
            
            inters = torch.stack(inters)
                
            pred = self.classification_model(inters)
            pred = (pred > 0).int().squeeze()
            
            return pred

        
    def __define_models(self, fasterrcnn_path, basicmodel_path, lstmmodel_path):
        
        fasterrcnn_path = Path(fasterrcnn_path)
        
        if isinstance(basicmodel_path, type(None)) and isinstance(lstmmodel_path, type(None)):
            raise ValueError("You must provde one of lstmmodel_path and basicmodel_path!")
        
        if not isinstance(basicmodel_path, type(None)) and not isinstance(lstmmodel_path, type(None)):
            raise ValueError("You can only define basicmodel_path or lstmmodel_path, not both!")
        
        if basicmodel_path:
            classification_path = Path(basicmodel_path)
        elif lstmmodel_path:
            classification_path = Path(lstmmodel_path)
    
        if fasterrcnn_path.is_file():
            box_model = model_from_file(str(fasterrcnn_path))
        elif fasterrcnn_path.is_dir():
            box_model = get_top_model(str(fasterrcnn_path))
        
        if classification_path.is_file():
            classification_model = model_from_file(str(classification_path))
        elif classification_path.is_dir():
            classification_model = get_top_model(str(classification_path))   
            
        return classification_model, box_model