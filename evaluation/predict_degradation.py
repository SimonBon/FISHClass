from tqdm import tqdm
import torch
from FISHClass.datasets import MYCN
from torch.utils.data import DataLoader
import numpy as np
import h5py
import os

def save_h5(degration_dict, path):
    
    with h5py.File(path, "w") as fout:
        
        for k, item in degration_dict.items():
        
            group = fout.create_group(k)
            group.create_dataset("X", data=item["X"])
            group.create_dataset("PRED", data=item["PRED"])
            group.create_dataset("TARGET", data=item["TARGET"])

def predict_degradation(model, dataset_path, device="cuda", dataset_kwargs=None, batch_size=32, verbose=True, save2h5=False, save_path=None):
    
    if isinstance(dataset_kwargs, type(None)):
        raise ValueError("Please probvide keyword arguments for MYCN Dataset")

    model.to(device)
    try:
        model.redefine_device(device)
    except:
        print("couldnt redefine")
        
    model.eval()
    
    if verbose:
        print(f"Using {device} for calculation")
        
    with h5py.File(dataset_path, "r") as fin:
        degradations = list(fin.keys())
        
    print(degradations)
        
    degration_dict = {}
    for degradation in degradations:

        dataset = MYCN(dataset_path, dataset=degradation, **dataset_kwargs)
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=8)
   
        preds, targets = torch.empty(0), torch.empty(0)
        
        if dataset.double_return:

            for X, X2, y in tqdm(dataloader):
                
                ŷ = model(X.to(device), X2.to(device)).cpu().detach()
                if ŷ.ndim == 0:
                    ŷ = ŷ.unsqueeze(0)
                ŷ = ŷ > 0
                preds = torch.cat((preds, ŷ))
                targets = torch.cat((targets, y))
                        
        else:
            
            for X, y in tqdm(dataloader):
            
                ŷ = model(X.to(device)).cpu().detach()
                if ŷ.ndim == 0:
                    ŷ = ŷ.unsqueeze(0)
                ŷ = ŷ > 0
                preds = torch.cat((preds, ŷ))
                targets = torch.cat((targets, y))

                    
        preds = np.array(preds).squeeze()
        targets = np.array(targets).squeeze()   
    
        if save2h5:
            
            h5_dict = {}
            with h5py.File(dataset_path, "r") as fin:

                h5_dict["X"] = np.array(fin[degradation]["X"])
                
            h5_dict["PRED"] = np.expand_dims(preds.astype(int),1) 
            h5_dict["TARGET"] = np.expand_dims(targets.astype(int),1) 
            
            degration_dict[degradation] = h5_dict
            
        else:
            degration_dict[degradation] = {"preds": preds, "targets": targets}  

    if save2h5:
        save_h5(degration_dict, os.path.join(save_path))

    return preds, targets


def predict_degradation_baseline(model, dataset_path, dataset_kwargs=None, batch_size=32, save2h5=False, save_path=None):
    
    if isinstance(dataset_kwargs, type(None)):
        raise ValueError("Please probvide keyword arguments for MYCN Dataset")

    with h5py.File(dataset_path, "r") as fin:
        degradations = list(fin.keys())
        
    degration_dict = {}
    for degradation in degradations:

        dataset = MYCN(dataset_path, dataset=degradation, **dataset_kwargs)
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=8)
   
        preds, targets = torch.empty(0), torch.empty(0)
        
        for X, y in tqdm(dataloader):
            
            ŷ = model(X) 
            if ŷ.ndim == 0:
                ŷ = ŷ.unsqueeze(0)

            preds = torch.cat((preds, ŷ))
            targets = torch.cat((targets, y))
            
        preds = np.array(preds).squeeze().astype(int)
        targets = np.array(targets).squeeze().astype(int)  
    
        if save2h5:
            
            h5_dict = {}
            with h5py.File(dataset_path, "r") as fin:

                h5_dict["X"] = np.array(fin[degradation]["X"])
                
            h5_dict["PRED"] = np.expand_dims(preds.astype(int),1) 
            h5_dict["TARGET"] = np.expand_dims(targets.astype(int),1) 
            
            degration_dict[degradation] = h5_dict
            
        else:
            degration_dict[degradation] = {"preds": preds, "targets": targets}  

    if save2h5:
        save_h5(degration_dict, os.path.join(save_path))

    return preds, targets