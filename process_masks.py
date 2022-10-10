import numpy as np
from tqdm import tqdm
import cv2
from CellClass.MCImage import MCImage

def get_cell_patches(MCIm: MCImage, masks: np.ndarray, channels=["B", "G", "R"], size=64):
    
    centers = get_cell_centers(masks)
    
    patches = extract_patches(MCIm, masks, centers, size, channels)
    
    return patches

        
def get_cell_centers(mask: np.ndarray) -> np.ndarray:
    
    centers = []  
    for n in tqdm(range(1,mask.max()+1)):
        tmp = np.copy(mask)
        tmp[mask != n] = 0
        tmp = tmp.astype(float)
        y, x = np.where(tmp != 0)
        y_min, y_max = y.min(), y.max()+1
        x_min, x_max = x.min(), x.max()+1
        
        if y_min == 0 or y_max == mask.shape[0] or x_min==0 or x_max==mask.shape[1]:
            continue
        
        cell = tmp[y_min:y_max, x_min:x_max]
        y, x = calc_center(cell)
        y_center, x_center = np.round(y+y_min,0), np.round(x+x_min,0)
        centers.append([y_center, x_center, n])
        
    return np.array(centers).astype(np.uint16)
        
def calc_center(bin):
    
    M = cv2.moments(bin)
    
    return M["m10"]/M["m00"], M["m01"]/M["m00"]

def extract_patches(MCIm, masks, centers, size, channels):
    
    if len(channels) == 1:
        im = getattr(MCIm, channels[0])
    else:
        im = np.stack([getattr(MCIm, x) for x in channels], axis=-1)
        
    if im.ndim == 2:
        tmp_im = np.pad(im, ((size//2, size//2),(size//2, size//2)), mode="constant")
        
    elif im.ndim == 3:
        tmp_im = np.pad(im, ((size//2, size//2),(size//2, size//2), (0,0)), mode="constant")
        
    tmp_masks = np.pad(masks, ((size//2, size//2),(size//2, size//2)), mode="constant")
    
    patches = []
    for y,x,n in tqdm(centers):
        
        y += size//2; x += size//2
        w_y, w_x = (y-size//2, y+size//2),(x-size//2, x+size//2)

        cell_mask = np.copy(tmp_masks[w_y[0]:w_y[1], w_x[0]:w_x[1]])
        cell_mask[cell_mask != n] = 0
        cell_mask = cell_mask.astype(bool)
        #cell_mask = dilate_mask(cell_mask, 7)
        
        
        marker_im = np.copy(tmp_im[w_y[0]:w_y[1], w_x[0]:w_x[1], ...])
        marker_all = np.copy(marker_im)
        marker_im[cell_mask == 0] = 0
        
        if cell_mask.any():
            #patch = Patch(cell_mask, marker_im, marker_all, channels, y, x, n)
            patch = Patch(cell_mask, marker_all, channels, y, x, n)
            patches.append(patch)
        
    return patches
        
def dilate_mask(mask, s=11):
    
    k = np.ones((s,s)).astype(np.uint8)
    ret = cv2.dilate(mask.astype(np.uint8), k)
    return ret.astype(bool)


class Patch():
    
    def __init__(self, mask, masked, channels, y_pos, x_pos, idx):
          
        for n, c in enumerate(channels):
            setattr(self, c, masked[..., n])
           
        if len(channels) == 3:
            self.RGB = np.stack((self.R, self.G, self.B), axis=-1)
            
        self.shape = masked.shape         
        self.mask = mask
        self.y_pos = y_pos
        self.x_pos = x_pos
        self.idx = idx
        
        self.y_size ,self.x_size = self.get_size()
        self.area = np.sum(mask)
        
        self.intensity_features()

    def get_size(self):
        
        y, x = np.where(self.mask != 0)
        
        y_min, y_max = y.min(), y.max()+1
        x_min, x_max = x.min(), x.max()+1
            
        return y_max-y_min, x_max-x_min
    
    
    def intensity_features(self):
        
        self.Features = Features()
        for c in ["R", "G", "B"]:
            tmp = getattr(self, c)[self.mask].ravel()
            
            for p in [2, 10, 25, 50, 75, 90, 98]:
                setattr(self.Features, f"intensity_{c}{p}", np.percentile(tmp, p))
            

class Features():
    
    def __init__(self):
        pass

    def __iter__(self):
        self._n = 0
        self._attrs = [self.__dict__[x] for x in self.__dict__ if not x.startswith("_")]
        return self
    
    def __next__(self):
        if self._n < len(self._attrs):
            self._n+=1
            return self._attrs[self._n-1]
        else:
            raise StopIteration
        