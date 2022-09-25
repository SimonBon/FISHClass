import h5py
import os
import numpy as np
from pathlib import Path
from xml.etree import ElementTree as ET
import pandas as pd
import cv2
import shutil
import random
from torch.utils.data import DataLoader
import torch
from tqdm import tqdm

from .evaluation import get_top_model
from FISHClass.datasets import MYCN
from .visualize import bbox_on_image
from .class_names import CLASS_NAMES
from .device import best_gpu

import matplotlib.pyplot as plt

def filelist(root, file_type):

    return  [Path(os.path.join(directory_path, f)) for directory_path, _, 
            files in os.walk(root) for f in files if f.endswith(file_type)]

def _match_lists(xml_list, png_list):
    
    xml_files, png_files = [], []
     
    png_samples = []
    for file in png_list:
        png_samples.append(file.stem)  
        
    for file in xml_list:
        xml_files.append(file)
        png_files.append(png_list[png_samples.index(file.stem)])
        
    return xml_files, png_files
  
  
def _bbox_from_xml(file):
    
    root = ET.parse(file).getroot()
    annotations = root.findall("./object/name")
    bboxes = root.findall("./object/bndbox")
    bbox_dict = []
    for anno, box in zip(annotations,bboxes):
        ymin = int(box.find("ymin").text)
        ymax = int(box.find("ymax").text)
        xmin = int(box.find("xmin").text)
        xmax = int(box.find("xmax").text)
        bbox_dict.append({"type": anno.text,
                            "box": [xmin, ymin, xmax, ymax]})

    return bbox_dict  


def _get_annotations_dataframe(xml_files, png_files):
    
    annotations = []
    for xml_file, png_file in zip(xml_files, png_files):
        
        boxes = _bbox_from_xml(xml_file)
        
        for i, inst in enumerate(boxes):
            anno_dict = {}
            anno_dict["filename"] = png_file.resolve()
            anno_dict["idx"] = i
            anno_dict["class"] = inst["type"]
            anno_dict["box"] = inst["box"]
            annotations.append(anno_dict)
    
    annotation_df = pd.DataFrame(annotations)
    return annotation_df

def _prepare_h5_dict(df):
    
    h5_dict = {}
    for filename in df.filename.unique():
        im = (cv2.cvtColor(cv2.imread(str(filename)), cv2.COLOR_BGR2RGB)/255).astype(float)
        tmp = df[df.filename == filename]
        
        labels, boxes = [], []
        for _, instance in tmp.iterrows():
            for n, class_name in enumerate(CLASS_NAMES):
                if instance["class"] == class_name:
                    labels.append(n+1)
                
            boxes.append(instance.box)
            
        h5_dict[str(filename.stem)] = {"labels": labels,
                                "boxes": np.array(boxes),
                                "image": np.array(im)}

    return h5_dict


def _create_h5_file(h5_dict, out_name, split):
    
    idxs = list(h5_dict.keys())
    n_train = int(split[0]*len(idxs))
    random.shuffle(idxs)
    idxs_train = idxs[:n_train]
    idxs_val = idxs[n_train:]

    with h5py.File(out_name, "w") as fin:
    
        fin.create_group("train")
        fin.create_group("val")
    
        keys = list(h5_dict.keys())
        random.shuffle(keys)
        
        for key in keys:
    
            items = h5_dict[key]
            if key in idxs_train:
                group = fin.create_group(f"train/{key}")
                
            else:
                group = fin.create_group(f"val/{key}")
                
            group.create_dataset("image", data=items["image"])
            group.create_dataset("labels", data=items["labels"])
            group.create_dataset("boxes", data=items["boxes"])
    

def create_h5_training(root, out_name, split=[0.9, 0.1]):
    
    xml_files = filelist(root, ".xml")
    png_files = filelist(root, ".png")
    xml_files, png_files = _match_lists(xml_files, png_files)
    
    annotation_df = _get_annotations_dataframe(xml_files, png_files)
    h5_dict = _prepare_h5_dict(annotation_df)
    
    
    _create_h5_file(h5_dict, out_name, split)
    
    
def create_annotation_images(base, out, n, sample=None):
    
    files = filelist(base, "png")
    random.shuffle(files)
    if isinstance(sample, str):
        files = [x for x in files if sample in x.stem]
    else:
        files = [x for x in files if "S19" in x.stem or "S29" in x.stem]
    
    i = 0
    for file in files:
        if os.path.isfile(os.path.join(out, file.name)):
            continue
        
        else:
            print("copied", os.path.join(out, file.name))
            image = cv2.cvtColor(cv2.imread(str(file)), cv2.COLOR_BGR2RGB)
            cv2.imwrite(os.path.join(out, str(file.name)), image)
            i+=1
            if i == n:
                break
    
def collate(batch):
    
    ims = []
    targets = []
    for b in batch:
        ims.append(b["image"].type(torch.FloatTensor))
        targets.append({"boxes": b["boxes"],
                        "labels": b["labels"]})
        
    return ims , targets

def get_bbox_prediction(fasterrcnn_model, image, target=None, ret_bbox_image=False, device="cuda", bbox_im_threshold=0):
    
    fasterrcnn_model.eval()
    fasterrcnn_model.to(device)
    
    out = fasterrcnn_model([image.to(device)])[0]
    
    res = {}
    res["boxes"] = out["boxes"].detach().cpu().numpy()
    res["labels"] = out["labels"].unsqueeze(1).detach().cpu().numpy()
    res["scores"] = out["scores"].unsqueeze(1).detach().cpu().numpy()
    res["image"] = image
    
    if not isinstance(target, type(None)):
        res["target"] = target
        
    if ret_bbox_image:
        bbox_image = bbox_on_image(res, threshold=bbox_im_threshold, ret=True)
        return res, bbox_image
     
    return res
    
    
def out2np(frcnn_output, device="cpu"):
    
    rets = []
    for res in frcnn_output:
        
        b = res["boxes"]
        l = res["labels"].unsqueeze(1)
        s = res["scores"].unsqueeze(1)
        
        pad = torch.zeros((100-b.shape[0], 6))
        
        ret = torch.cat((b, l, s), axis=1)
        rets.append(torch.cat((ret, pad.to(device)), axis=0))
    
    return torch.stack(rets)


def get_train_val_test_idxs(n_samples, split=[0.8, 0.1, 0.1]):
    
    idxs = list(range(n_samples))
    random.shuffle(idxs)
    
    n_train = int(n_samples*split[0])
    n_val = int(n_samples*split[1])
    
    train_idxs = idxs[:n_train]
    val_idxs = idxs[n_train:n_train+n_val]
    test_idxs = idxs[n_train+n_val:]
    
    return np.array(train_idxs).astype(int), np.array(val_idxs).astype(int), np.array(test_idxs).astype(int)
    

def create_boxes_h5(fasterrcnn_path, out_name, device=None, h5_in_file="/data_isilon_main/isilon_images/10_MetaSystems/MetaSystemsData/MYCN_SpikeIn/results/h5/trainset_cleaned_small.h5", n=[None, None, None]):
    
    if isinstance(device, type(None)):
        device = best_gpu()
      
    model = get_top_model(fasterrcnn_path)
    model.to(device)
    model.eval()
    
    train_set = MYCN(h5_in_file, dataset="train", norm_type=None, transform=None, n=n[0])
    val_set = MYCN(h5_in_file, dataset="val", norm_type=None, transform=None,  n=n[1])
    test_set = MYCN(h5_in_file, dataset="test", norm_type=None, transform=None,  n=n[2])
    
    train_loader = DataLoader(train_set, batch_size=32, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_set, batch_size=32, shuffle=True, num_workers=2)
    test_loader = DataLoader(test_set, batch_size=32, shuffle=True, num_workers=2)
    
    data = {}
    for set_name, loader in zip(["train", "val", "test"], [train_loader, val_loader, test_loader]):
        X_tmp, y_tmp = [], []
        for X, y in tqdm(loader):
            
            y_tmp.extend(y.numpy())
            X = [{k: v.detach().cpu() for k, v in result.items()} for result in model(X.to(device))]
            X_tmp.append(out2np(X).numpy())
            
        X_tmp = np.vstack(X_tmp)
        y_tmp = np.array(y_tmp)
        
        data[set_name] = {"X": X_tmp, "y": y_tmp}
        
    with h5py.File(out_name, "w") as fout:
        
        for key, item in data.items():
            
            fout.create_group(key)
            fout.create_dataset(f"{key}/X", data=item["X"])
            fout.create_dataset(f"{key}/y", data=item["y"])