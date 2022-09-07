import torch

def train_fn(self, train_loader, loss_fn, optimizer, scheduler, device):
    
    train_loss = 0
    for X, y in train_loader:
        
        X = [x.to(device) for x in X]
        y = [{k: v.to(device) for k,v in sub_y.items()} for sub_y in y]
        
        ŷ = self.forward(X, y)
        loss = loss_fn(val for val in ŷ.values())
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if not isinstance(scheduler, type(None)):
            scheduler.step()
        
        train_loss += loss.item()
        del ŷ, X, y, loss
        
    train_loss = train_loss/len(train_loader)
    ret = {"train_loss": train_loss}
        
    return ret

def validation_fn(self, validation_loader, loss_fn, device):
    
    val_loss = 0
    with torch.no_grad():

        for X, y in validation_loader:
            
            X = [im.to(device) for im in X]
            y = [{k: v.to(device) for k,v in sub_y.items()} for sub_y in y]

            out = self.forward(X, y)
            
            val_loss += loss_fn(loss for loss in out.values()).item()
            del out, X, y

    val_loss = val_loss/len(validation_loader)
    ret_dict = {"val_loss": val_loss}
    return ret_dict