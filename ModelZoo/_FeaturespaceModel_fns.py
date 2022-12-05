import torch
from tqdm import tqdm

def train_fn(self, train_loader, loss_fn, optimizer, scheduler, device):
    
    train_loss = 0
    for X, X2, y in tqdm(train_loader):
        
        if self.custom_loss:
            
            ŷ, fs = self.forward(X.to(device), X2.to(device), return_box_fs=True)
            ŷ = ŷ.squeeze()
            fs = fs.reshape(-1, 100, 6)

            sizes = torch.tensor([get_cell_bbox(patch) for patch in X2])
            areas = torch.tensor([get_bbox_size(patch_fs) for patch_fs in fs])
            
            areas = torch.clip((areas/sizes), 0, 1)
            loss = loss_fn(ŷ, y.to(device), areas.to(device))
        
        else:
            
            ŷ = self.forward(X.to(device), X2.to(device)).squeeze()
            loss = loss_fn(ŷ, y.to(device))
            
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if not isinstance(scheduler, type(None)):
            scheduler.step()
        
        train_loss += loss.item()
        
    train_loss = train_loss/len(train_loader)
    ret = {"train_loss": train_loss}
        
    return ret


def validation_fn(self, validation_loader, loss_fn, device):
        
    self.eval()
    
    val_loss = 0
    accuracy, n = 0, 0
    with torch.no_grad():

        for X, X2, y in tqdm(validation_loader):
            
            y = y.squeeze()
            n += len(y)
            
            if self.custom_loss:
                
                ŷ, fs = self.forward(X.to(device), X2.to(device), return_box_fs=True)
                ŷ = ŷ.squeeze()
                fs = fs.reshape(-1, 100, 6)

                sizes = torch.tensor([get_cell_bbox(patch) for patch in X2])
                areas = torch.tensor([get_bbox_size(patch_fs) for patch_fs in fs])
                
                areas = torch.clip((areas/sizes), 0, 1)
                
                val_loss += loss_fn(ŷ, y.to(device), areas.to(device)).item()
                
            else:
                
                ŷ = self.forward(X.to(device), X2.to(device)).squeeze()
                val_loss += loss_fn(ŷ, y.to(device)).item()
                
            accuracy += sum((ŷ>0) == y.to(device))

    val_loss = val_loss/len(validation_loader)
    ret_dict = {"val_loss": val_loss,
                "accuracy": (accuracy/n).item()*100}
    
    self.train() 
    
    return ret_dict 


def get_cell_bbox(patch):
    
    A, B = torch.where(patch[2]!=0)
    sz = (A.max()-A.min())*(B.max()-B.min())
    return sz

def get_bbox_size(patch_fs):
    
    x0, y0, x1, y1 = patch_fs[:,0], patch_fs[:,1], patch_fs[:,2], patch_fs[:,3]
    c = patch_fs[:,4]
    s = patch_fs[:,5]
    
    area = torch.sum((x1-x0) * (y1-y0)*s)

    return area