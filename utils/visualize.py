import numpy as np
import cv2
import matplotlib.pyplot as plt
import pandas as pd

def bbox_on_image(data_dict, sz=(400,400), ret=False, threshold=None):
    
    im = (np.array(data_dict["image"]*255).transpose(1,2,0)).astype(np.uint8).copy()
    boxes = np.array(data_dict["boxes"]).astype(int)
    labels = np.array(data_dict["labels"]).astype(int)
    
    if isinstance(threshold, float):
        scores = np.array(data_dict["scores"]).astype(float)
        idxs = (scores > threshold).squeeze()
        boxes = boxes[idxs]
        labels = labels[scores > threshold]
        
    for bbox, label in zip(boxes, labels):
    
        add_bbox(im, bbox, label)
        
    if ret:
        return im
            
    plt.imshow(im)  
    plt.show()
        
def add_bbox(im, bbox, label):
    
    try:
        if label == 1:
            cv2.rectangle(im, (bbox[0],bbox[1]), (bbox[2],bbox[3]), color=(255,0,0))
        elif label == 2:
            cv2.rectangle(im, (bbox[0],bbox[1]), (bbox[2],bbox[3]), color=(0,255,0))
        elif label == 3:
            cv2.rectangle(im, (bbox[0],bbox[1]), (bbox[2],bbox[3]), color=(0,0,255))
    except:
        pass
        
        
def plot_results(results_list, return_image=False):
    
    results_df = pd.DataFrame(results_list)
    
    fig, ax = plt.subplots()
    
    for val, items in results_df.items():
        
        if val == "accuracy":
            acc = ax.plot([], label="accuracy")
            ax2 = ax.twinx()
            ax2.plot(items, color=acc[0].get_color())
            #ax2.set_ylim(0,1.1)
            
        else:
            ax.plot(items)
        
    ax.set_yscale("log")
    ax.legend(results_df.columns)
    
    if return_image:
        return fig
    else:
        plt.show()