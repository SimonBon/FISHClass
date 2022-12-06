import os
import pickle as pkl
import numpy as np
import cv2
from tqdm import tqdm
import random
import argparse
from pathlib import Path
import h5py
from FISHClass.segmentation import Segmentation
from FISHClass.MCImage import MCImage
from FISHClass.process_masks import get_cell_patches

import matplotlib.pyplot as plt


SIZE = (1496, 2048)

def parse():
    
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--in_dir", type=str)
    parser.add_argument("-s", "--sample", type=str)
    parser.add_argument("-o", "--out_dir", type=str)
    parser.add_argument("--n_cellline_images", type=int, default=None)
    parser.add_argument("--algorithm", type=str, default="cellpose")
    parser.add_argument("--diameter", type=int, default=30)
    parser.add_argument("--target_value", type=int, default=-1)
    parser.add_argument("--n_patches", type=int, default=None)
    
    return parser.parse_args()


def define_out_dir(out_dir, sample):
    
    if not os.path.isdir(out_dir):
        decision = input(f"{out_dir} does not exist, do you want to create it? [y/n]")
        if decision.lower() == "y":
            os.makedirs(out_dir)
            print(f"{out_dir} created!")
            
        else:
            print(f"{out_dir} not created!")
            exit()
                
    n_out_dir = os.path.join(out_dir, sample)
    if not os.path.isdir(n_out_dir):
        os.makedirs(n_out_dir)
        print(f"{n_out_dir} created!")
    return n_out_dir


def get_image_paths(in_dir, n=None):
    
    files = os.listdir(in_dir)
    files = [x for x in files if "b.tif" in x.lower()]
    random.shuffle(files)
    files = files[:n]
    return files

def get_RGB_images(in_dir, image_paths):

    RGBs = {}
    for file in tqdm(image_paths):
        
        n = file.split("-")[1]
        R = cv2.imread(os.path.join(in_dir, f"Img-{n}-R.TIF"),0)
        G = cv2.imread(os.path.join(in_dir, f"Img-{n}-G.TIF"),0)
        B = cv2.imread(os.path.join(in_dir, f"Img-{n}-B.TIF"),0)
        
        R = (R/R.max()*255).astype(np.uint8)
        G = (G/G.max()*255).astype(np.uint8)
        B = (B/B.max()*255).astype(np.uint8)

        RGB = np.stack((R,G,B)).transpose(1,2,0)
        
        if RGB.shape[:2] != SIZE:
            RGB =cv2.resize(RGB, dsize=(SIZE[1], SIZE[0]))
            
        RGBs[int(n)] = RGB
        
    return RGBs

if __name__ == "__main__":
    
    args = parse()
    
    sample_dir = define_out_dir(args.out_dir, args.sample)
    image_paths = get_image_paths(args.in_dir, n=args.n_cellline_images)

    RGB_images = get_RGB_images(args.in_dir, image_paths)
    
    S = Segmentation(args.algorithm)

    # for each sample extract the patches
    for dir_name in ["overlays", "masks"]:
        if not os.path.isdir(os.path.join(sample_dir, dir_name)):
            os.mkdir(os.path.join(sample_dir, dir_name))
      
    patches = []  
    for i, image in RGB_images.items():

        MCIm = MCImage(image, scheme="RGB")
        MCIm.normalize()

        if args.algorithm == "deepcell":
            im, res, o = S(MCIm.B, return_outline=True, image_mpp=0.4)
            
        elif args.algorithm == "cellpose":
            im, res, o = S(MCIm.B, return_outline=True, diameter=args.diameter)

        cv2.imwrite(os.path.join(sample_dir, f"overlays/overlay_{i}.png"), o*255)
        cv2.imwrite(os.path.join(sample_dir, f"masks/mask_{i}.tif"), o.astype(np.uint16))
    
        if res.max() < 25:
            continue
        patches.extend(get_cell_patches(MCIm, res, size=128))

    random.shuffle(patches)
    patches = patches[:args.n_patches]

    ims = []
    for patch in tqdm(patches):
            
        im = np.copy(patch.RGB)
        im[~patch.mask]=0
        ims.append(im.copy())
        
    ims = np.array(ims)
    
    with h5py.File(os.path.join(sample_dir, f"{args.sample}.h5"), "w") as fout:
        
        group = fout.create_group(args.sample)
        group.create_dataset("X", data=ims.astype(np.float32))
        group.create_dataset("y", data=(args.target_value*np.ones(ims.shape[0])).astype(np.float32))