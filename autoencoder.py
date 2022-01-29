import numpy as np
import torch
import torchvision
import matplotlib.pyplot as plt
import copy
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from skimage import io, transform
from torchvision import transforms, utils
import os
from tqdm import tqdm


class MNIST_Encoder(nn.Module):
    def __init__(self, input_size, encoding_dim):
        super().__init__()
        self.layers = nn.ModuleList(
            [nn.Linear(input_size, encoding_dim),
             nn.ReLU()]
        )

    def forward(self,x):
        for layer in self.layers:
            x = layer(x)
        return x
    
class MNIST_Decoder(nn.Module):
    def __init__(self, encoding_dim, input_size):
        super().__init__()
        self.layers = nn.ModuleList(
                [nn.Linear(encoding_dim, input_size)]
        )
        
    def forward(self,z):
        for layer in self.layers:
            z = layer(z)
        return z

class MNIST_Autoencoder(nn.Module):
    def __init__(self, input_size = 784, encoding_dim = 32):
        super().__init__()
        self.encoder = MNIST_Encoder(input_size, encoding_dim)
        self.decoder = MNIST_Decoder(encoding_dim, input_size)
    
    def forward(self, x):
        x = x.view(x.size(0), -1) # flatten to (nm, 1) vector
        x = self.encoder(x) # here we get the latent z
        x = self.decoder(x) # here we get the reconsturcted input
        x = torch.sigmoid(x)
        x = x.reshape(x.size(0), 1, 28, 28) # reshape this flatten vector to the original image size    
        return x
    

    

    
class Encoder(nn.Module):
    def __init__(self, input_size, encoding_dim):
        super().__init__()
        self.layers = nn.ModuleList(
            [nn.Linear(input_size, 4096),
            nn.ReLU(),
            nn.Linear(4096, encoding_dim),
            nn.ReLU()])

    def forward(self,x):
        for layer in self.layers:
            x = layer(x)
        return x
    
class Decoder(nn.Module):
    def __init__(self, encoding_dim, input_size):
        super().__init__()
        
        self.layers = nn.ModuleList(
            [nn.Linear(encoding_dim, 4096),
            nn.ReLU(), 
            nn.Linear(4096, input_size)])
    
    def forward(self,z):
        for layer in self.layers:
            z = layer(z)
        return z
    
class Autoencoder(nn.Module):
    def __init__(self, input_size = 224*224, encoding_dim = 1024):
        super().__init__()
        self.encoder = Encoder(input_size, encoding_dim)
        self.decoder = Decoder(encoding_dim, input_size)
    
    def forward(self, x):
        x = x.view(x.size(0), -1) # flatten to (nm, 1) vector
        x = self.encoder(x) # here we get the latent z
        x = self.decoder(x) # here we get the reconsturcted input
        x = torch.sigmoid(x)
        x = x.reshape(x.size(0), 1, 224, 224) # reshape this flatten vector to the original image size    
        return x
    
    
    
    
def train(model, trainloader = None, valloader = None, num_epochs = 1):
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    
    # name dataloaders for phrase
    phases = ['train']
    dataloaders = {'train':trainloader}
    if valloader:
        phases.append('valid')
        dataloaders['valid'] = valloader
        
    model.to(device)
    optimizer = torch.optim.Adadelta(model.parameters())
    #criterion = F.binary_cross_entropy(autoencoder(x), target)
    criterion = torch.nn.BCELoss()
    
    for epoch in range(num_epochs):
        print(f'Epoch {epoch+1}/{num_epochs}\n{"-"*10}')
        
        for phase in phases:
            if phase == 'train':
                model.train()
            else:
                model.eval()
                
            running_loss, running_correct, count = 0.0, 0, 0
            for batch_idx, x in tqdm(enumerate(dataloaders[phase])):
            #for batch_idx, (x, y) in enumerate(dataloaders[phase]):
                #print(f"Batch {batch_idx}")
                #x,y = x.to(device), y.to(device)
                x = x.to(device)

                # zero param gradients
                optimizer.zero_grad()

                # forward: track history if training phase
                with torch.set_grad_enabled(phase=='train'): # pytorch >= 0.4
                    outputs = model(x)
                    loss    = criterion(outputs, x)
                    #preds,_ = torch.max(outputs,1) # for accuracy metric
                    # backward & optimize if training phase
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()
                
                # stats
                running_loss += loss.item() * x.size(0)
                count += len(x)
            
            epoch_loss = running_loss / count
            print(f'{phase} loss {epoch_loss:.6f}')
        print()
            