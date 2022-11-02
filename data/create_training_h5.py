import h5py
import os
import cv2
import numpy as np
import argparse
from pathlib import Path
from tqdm import tqdm
from FISHClass.utils.data import get_train_val_test_idxs
import random
import h5py 
import torch

def parse():
    
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--in_file", type=str, required=True, help="enter the path to your .pt file")
    parser.add_argument("-o", "--out_file", type=str, required=True, help="enter the path and filename you want to use as output")
    return parser.parse_args()

if __name__ == "__main__":
    
    args = parse()

    sample_dict = torch.load(args.in_file)
        
    positive = sample_dict["positive"]
    negative = sample_dict["negative"]
    
    n = min([positive.shape[0], negative.shape[0]])
    
    positive = positive[:n]
    pos_targets = np.ones(positive.shape[0])
    negative = negative[:n]
    neg_targets = np.zeros(negative.shape[0])

    images = np.concatenate((positive, negative))
    targets = np.concatenate((pos_targets, neg_targets))
    
    train_idx, val_idx, test_idx = get_train_val_test_idxs(len(targets))
    
    train_X = images[train_idx]
    train_y = targets[train_idx]
    val_X = images[val_idx]
    val_y = targets[val_idx]
    test_X = images[test_idx]
    test_y = targets[test_idx]
    
    with h5py.File(args.out_file, "w") as fout:
        
        group = fout.create_group("train")
        group.create_dataset("X", data=train_X.astype(np.float32))
        group.create_dataset("y", data=train_y.astype(np.float32))
        group = fout.create_group("val")
        group.create_dataset("X", data=val_X.astype(np.float32))
        group.create_dataset("y", data=val_y.astype(np.float32))
        group = fout.create_group("test")
        group.create_dataset("X", data=test_X.astype(np.float32))
        group.create_dataset("y", data=test_y.astype(np.float32))


    print("Saved under: ", args.out_file)

    
    
    