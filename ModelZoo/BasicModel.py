import torch
from torch import nn

#test comment

class BasicClassifier(nn.Module):
    
    def __init__(self, input_size, layers = [128, 128, 128], output_size=1, drop_p=0.5):
        
        self.input_size = input_size
        self.layers = layers
        self.output_size = output_size
        
        self.drop_p = drop_p
        
        self.kwargs = {k: v for k, v in self.__dict__.items()}
        
        super().__init__()
    
        self.sizes = [input_size, *layers, output_size]
    
        self.fc =  nn.ModuleList([self.Layer(self.sizes[i], self.sizes[i+1], self.drop_p) for i in range(len(self.sizes)-1)])
        
        
    def forward(self, X):

        for layer in self.fc:
            X = layer(X)
    
        return X.squeeze()
    
    
    class Layer(nn.Module):
        
        def __init__(self, input_size, out_size, drop_p):
            super().__init__()
            
            self.out_size = out_size
            self.fc = nn.Linear(input_size, out_size)
            if out_size != 1:
                self.dropout = nn.Dropout(drop_p)
                self.relu = nn.ReLU()
            
        def forward(self, X):
            
            X = X.flatten(start_dim=1)
            
            X = self.fc(X)
            
            if self.out_size != 1:
                X = self.dropout(X)
                X = self.relu(X)

            return X
        
        
    def train_fn(self, train_loader, loss_fn, optimizer, scheduler, device):
            
        train_loss = 0
        for X, y in train_loader:
            
            ŷ = self.forward(X.to(device))
            loss = loss_fn(ŷ, y.to(device))
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            if not isinstance(scheduler, type(None)):
                scheduler.step()
            
            train_loss += loss.item()
            
        train_loss /= len(train_loader)
        ret = {"train_loss": train_loss}
            
        return ret
    
    
    def validation_fn(self, validation_loader, loss_fn, device):
        
        self.eval()
        val_loss = 0
        accuracy, n = 0, 0
        with torch.no_grad():

            for X, y in validation_loader:
                
                n += len(y)
                ŷ = self.forward(X.to(device))
                val_loss += loss_fn(ŷ, y.to(device)).item()
                accuracy += sum((ŷ>0) == y.to(device))
                
        val_loss /= len(validation_loader)
        ret_dict = {"val_loss": val_loss,
                    "accuracy": (accuracy/n).item()*100}
        
        self.train() 
        
        return ret_dict