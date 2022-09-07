import nvidia_smi
import torch
import time
import random

def best_gpu():
    
    nvidia_smi.nvmlInit()

    max_free = 0
    for i in range(torch.cuda.device_count()):

        handle = nvidia_smi.nvmlDeviceGetHandleByIndex(i)

        info = nvidia_smi.nvmlDeviceGetMemoryInfo(handle)
        
        if info.free > max_free:
            idx = i
            max_free = info.free
            

    return torch.device(idx)