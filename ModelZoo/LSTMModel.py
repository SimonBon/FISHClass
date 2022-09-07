import torch
from torch import nn

class LSTMClassifier(nn.Module):
    
    def __init__(self, input_size, hidden_size, output_size=1, drop_p=0.3):
        
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        
        self.kwargs = {k: v for k, v in self.__dict__.items()}
        
        super().__init__()
        
        self.lstm = nn.LSTM(input_size=input_size, hidden_size=hidden_size, batch_first=True)
        
        self.fc =  nn.Sequential(
            nn.Linear(hidden_size, 100),
            nn.ReLU(),
            nn.Dropout(drop_p),
            nn.Linear(100, 100),
            nn.ReLU(),
            nn.Dropout(drop_p),
            nn.Linear(100, output_size)
        )
        
        
    def forward(self, X):

        X, _ = self.lstm(X)
        
        X = self.fc(X)
        X = X[:, -1]
        
        return X.squeeze()
    
    
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