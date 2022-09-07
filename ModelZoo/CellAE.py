import torch
from torch import nn

class ConvBlock(nn.Module):
    
    def __init__(self, in_channels, out_channels):
        super().__init__()
        
        if in_channels < out_channels:
            self.block = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(),
                nn.MaxPool2d(kernel_size=2))
        else:
            pass
            self.block = nn.Sequential(
                nn.ConvTranspose2d(in_channels, out_channels, kernel_size=3, stride=2, padding=1, output_padding=1),
                nn.BatchNorm2d(out_channels),
                nn.ReLU())
        
    def forward(self, X):
        
        return self.block(X)
    
    

class CellAE(nn.Module):
    
    def __init__(self, latent_size=8, inp_size=[128, 128], layers=[3, 8, 16, 32]):
        super().__init__()
        
        self.latent_size = latent_size
        self.inp_size = inp_size
        self.layers = layers
        self.encoder = self.create_encoder()
        self.decoder = self.create_decoder()
        self.to_latent_space = nn.Sequential(nn.Linear(int((self.inp_size[0])/(2**(len(self.layers)-1))*(self.inp_size[1])/(2**(len(self.layers)-1))*self.layers[-1]), 1000),
                                             nn.ReLU(),
                                             nn.Linear(1000, self.latent_size))
        
        self.from_latent_space = nn.Sequential(nn.Linear(self.latent_size, 1000), 
                                               nn.ReLU(),
                                               nn.Linear(1000, int((self.inp_size[0])/(2**(len(self.layers)-1))*(self.inp_size[1])/(2**(len(self.layers)-1))*self.layers[-1])))
          
    def create_encoder(self):
        return nn.Sequential(*[ConvBlock(self.layers[i], self.layers[i+1]) for i in range(len(self.layers)-1)])


    def create_decoder(self):
        return nn.Sequential(*[ConvBlock(self.layers[i], self.layers[i-1]) for i in range(len(self.layers)-1, 0, -1)])
    
    
    def forward(self, X):
        
        X = self.encoder(X)
        latent = X.view(X.shape[0], -1)
        latent_vals = self.to_latent_space(latent)
        latent = self.from_latent_space(latent_vals)
        X = latent.view(X.shape)
        X = self.decoder(X)

        return X, latent_vals
    
    
    def reconstruct(self, latent):
        
        latent = self.from_latent_space(latent)
        X = latent.view((latent.shape[0], self.layers[-1], int(self.inp_size[0]/2**(len(self.layers)-1)), int(self.inp_size[1]/2**(len(self.layers)-1))))
        X = self.decoder(X)
        return X
        
        
    def encode(self, X):
            
        X = self.encoder(X)
        latent = X.view(X.shape[0], -1)
        latent_vals = self.to_latent_space(latent)
        
        return latent_vals