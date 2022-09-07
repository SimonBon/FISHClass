import torch

def train_fn(self, train_loader, loss_fn, optimizer, scheduler, device):
    
    train_loss = 0
    for X, X2, y in train_loader:
        
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

        for X, X2, y in validation_loader:
            
            y = y.squeeze()
            n += len(y)
            ŷ = self.forward(X.to(device), X2.to(device)).squeeze()
            val_loss += loss_fn(ŷ, y.to(device)).item()
            accuracy += sum((ŷ>0) == y.to(device))

    val_loss = val_loss/len(validation_loader)
    ret_dict = {"val_loss": val_loss,
                "accuracy": (accuracy/n).item()*100}
    
    self.train() 
    
    return ret_dict 