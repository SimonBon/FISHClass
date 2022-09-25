from tqdm import tqdm
import torch
from FISHClass.datasets import MYCN
from torch.utils.data import DataLoader
import numpy as np
import h5py
import os

def save_h5(h5_dict, path):
    
    with h5py.File(path, "w") as fout:
        
            fout.create_dataset("X", data=h5_dict["X"])
            fout.create_dataset("PRED", data=h5_dict["PRED"])
            fout.create_dataset("TARGET", data=h5_dict["TARGET"])

def predict_test(model, dataset_path, device="cuda", dataset_kwargs=None, batch_size=32, verbose=True, save2h5=False, save_path=None):
    
    if isinstance(dataset_kwargs, type(None)):
        raise ValueError("Please probvide keyword arguments for MYCN Dataset")

    model.to(device)
    model.eval()
    
    if verbose:
        print(f"Using {device} for calculation")

    dataset = MYCN(dataset_path, dataset="test", **dataset_kwargs)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=8)
   
    preds, targets = torch.empty(0), torch.empty(0)
    
    if dataset.double_return:

        model.redefine_device(device)
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

            h5_dict["X"] = np.array(fin["test"]["X"])
                        
        h5_dict["PRED"] = np.expand_dims(preds.astype(int),1) 
        h5_dict["TARGET"] = np.expand_dims(targets.astype(int),1) 

    if save2h5:
        save_h5(h5_dict, os.path.join(save_path))

    return preds, targets

def save_h5(h5_dict, path):
    
    with h5py.File(path, "w") as fout:
        
            fout.create_dataset("X", data=h5_dict["X"])
            fout.create_dataset("PRED", data=h5_dict["PRED"])
            fout.create_dataset("TARGET", data=h5_dict["TARGET"])

def predict_test_baseline(model, dataset_path, dataset_kwargs=None, batch_size=32, verbose=True, save2h5=False, save_path=None):
    
    if isinstance(dataset_kwargs, type(None)):
        raise ValueError("Please probvide keyword arguments for MYCN Dataset")
    

    dataset = MYCN(dataset_path, dataset="test", **dataset_kwargs)
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

            h5_dict["X"] = np.array(fin["test"]["X"])
                        
        h5_dict["PRED"] = np.expand_dims(preds.astype(int),1) 
        h5_dict["TARGET"] = np.expand_dims(targets.astype(int),1) 

    if save2h5:
        save_h5(h5_dict, os.path.join(save_path))

    return preds, targets