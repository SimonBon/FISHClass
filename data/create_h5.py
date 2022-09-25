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

def parse():
    
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--in_dir", type=str, required=True, help="enter the path to your directory containing .png files of single cells")
    parser.add_argument("-o", "--out_file", type=str, required=True, help="enter the path and filename you want to use as output")
    parser.add_argument("-m", "--mask", action="store_true")
    parser.add_argument("-g", "--only_green", action="store_true")
    parser.add_argument("-n", "--num", type=int)
    return parser.parse_args()

if __name__ == "__main__":
    
    args = parse()
    
    base = Path(args.in_dir)  
    files = list(base.glob("*.png"))

    ims, ts = [], []
    for file in tqdm(files):
        file = str(file)
        im = cv2.imread(file)
        
        if args.mask:
            
            mask = np.zeros_like(im[...,0])
            mask[im[...,2]!=0] = 255
            #mask = mask.astype(bool) wird evtl. noch benötigt
            
            im[...,2] = mask
            
            
        if args.only_green:
            
            im[..., 0] = np.zeros_like(im[..., 0])
            
        ims.append(im)
        ts.append(1) if "pos" in file else ts.append(0)
    
    tmp = list(zip(ims, ts))
    random.shuffle(tmp)
    ims, ts = zip(*tmp)
    
    n_ims = np.array(ims)
    t_ts = np.array(ts)
    
    if isinstance(args.num, type(None)):
        args.num = len(ts)//2
    else:
        args.num = args.num // 2

    pos = n_ims[t_ts==1][:args.num]
    neg = n_ims[t_ts==0][:args.num]
    pos_t = t_ts[t_ts==1][:args.num]
    neg_t = t_ts[t_ts==0][:args.num]

    images = np.concatenate((pos, neg))
    targets = np.concatenate((pos_t, neg_t))
    
    train_idx, val_idx, test_idx = get_train_val_test_idxs(len(targets))
    
    train_X = images[train_idx]
    train_y = targets[train_idx]
    val_X = images[val_idx]
    val_y = targets[val_idx]
    test_X = images[test_idx]
    test_y = targets[test_idx]
    
    with h5py.File(args.out_file, "w") as fout:
        group = fout.create_group("train")
        group.create_dataset("X", data=train_X)
        group.create_dataset("y", data=train_y)
        group = fout.create_group("val")
        group.create_dataset("X", data=val_X)
        group.create_dataset("y", data=val_y)
        group = fout.create_group("test")
        group.create_dataset("X", data=test_X)
        group.create_dataset("y", data=test_y)


    

    
    
    