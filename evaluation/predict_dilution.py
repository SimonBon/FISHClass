import torch
from FISHClass import models
from FISHClass.datasets import MYCN
from torch.utils.data import DataLoader
from tqdm import tqdm
import h5py
from baseline.models import SpotdetectionModel
import numpy as np
import os

def save_h5(h5_dict, path):
    
    with h5py.File(path, "w") as fout:
        
        for k, v in h5_dict.items():
            
            group = fout.create_group(k)
            group.create_dataset("X", data=v["X"])
            group.create_dataset("PRED", data=v["PRED"])

            
def predict_mixture(model, dataset_path, sample=None, device="cuda", dataset_kwargs=None, batch_size=32, n=500, trials=10, verbose=True, save2h5=False, save_path=None):
    
    if isinstance(dataset_kwargs, type(None)):
        raise ValueError("Please probvide keyword arguments for MYCN Dataset")

    if isinstance(sample, str):
        SAMPLES = [sample]
            
    elif isinstance(sample, list):
        SAMPLES = sample
        
    elif isinstance(sample, type(None)):

        with h5py.File(dataset_path, "r") as fin:
            SAMPLES = [x for x in fin.keys()]

    model.to(device)
    model.eval()

    results = {}
    
    if verbose:
        print(f"Samples evaluated: {SAMPLES}")
        print(f"Using {device} for calculation")

    h5_dict = {}
    for sample in SAMPLES:
        
        sample_results = []
        for _ in tqdm(range(trials)):

            dataset = MYCN(dataset_path, sample, **dataset_kwargs)
            dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=not(save2h5), num_workers=8)

            if isinstance(n, type(None)):
                n = len(dataset)

            if dataset.double_return:

                model.redefine_device(device)
                
                preds = torch.empty(0)
                for X, X2, _ in tqdm(dataloader):
                    
                    ŷ = model(X.to(device), X2.to(device)).cpu().detach()
                    if ŷ.ndim == 0:
                        ŷ = ŷ.unsqueeze(0)
                    ŷ = ŷ > 0
                    preds = torch.cat((preds, ŷ))

                    if len(preds) > n:
                        preds = preds[:n]
                        break
                                    
            else:
                
                preds = torch.empty(0)
                for X, _ in tqdm(dataloader):
                
                    ŷ = model(X.to(device)).cpu().detach()
                    if ŷ.ndim == 0:
                        ŷ = ŷ.unsqueeze(0)
                    ŷ = ŷ > 0
                    preds = torch.cat((preds, ŷ))
                        
                    if len(preds) > n:
                        preds = preds[:n]
                        break   
                        
            preds = np.array(preds).squeeze()
            sample_results.append(float(sum(preds)/n)*100)
                
        print(sample, sample_results)      
        results[sample] = sample_results
        
        if save2h5:

            with h5py.File(dataset_path, "r") as fin:

                h5_dict[sample] = {"X": np.array(fin[sample]["X"][:n])}
                            
            h5_dict[sample]["PRED"] = np.expand_dims(preds.astype(int),1)
        
    if save2h5:
            save_h5(h5_dict, os.path.join(save_path))
        
    return results


def predict_mixture_baseline(model, dataset_path, sample=None, dataset_kwargs=None, batch_size=32, n=500, trials=10, verbose=False, save2h5=False, save_path=None):
    
    if isinstance(dataset_kwargs, type(None)):
        raise ValueError("Please probvide keyword arguments for MYCN Dataset")

    if isinstance(sample, str):
        SAMPLES = [sample]
            
    elif isinstance(sample, list):
        SAMPLES = sample
        
    elif isinstance(sample, type(None)):

        with h5py.File(dataset_path, "r") as fin:
            SAMPLES = [x for x in fin.keys()]

    results = {}
    
    if verbose:
        print(f"Samples evaluated: {SAMPLES}")

    h5_dict = {}
    for sample in SAMPLES:
        

        sample_results = []
        for _ in tqdm(range(trials)):

            dataset = MYCN(dataset_path, sample, **dataset_kwargs)
            dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=not(save2h5), num_workers=8)

            if isinstance(n, type(None)):
                n = len(dataset)

            preds = torch.empty(0)
            for X, _ in tqdm(dataloader):
                
                ŷ = model(X) 
                if ŷ.ndim == 0:
                    ŷ = ŷ.unsqueeze(0)

                preds = torch.cat((preds, ŷ))
                
                if len(preds) > n:
                    preds = preds[:n]
                    break
            
            preds = np.array(preds).squeeze()
            sample_results.append(float(sum(preds[preds!=-1])/len(preds[preds!=-1]))*100)
                
        print(sample, sample_results)      
        results[sample] = sample_results
        
        if save2h5:

            with h5py.File(dataset_path, "r") as fin:

                h5_dict[sample] = {"X": np.array(fin[sample]["X"][:n])}
                            
            h5_dict[sample]["PRED"] = np.expand_dims(preds.astype(int),1)
        
    if save2h5:
        save_h5(h5_dict, os.path.join(save_path))
        
    return results
    