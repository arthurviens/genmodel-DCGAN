import numpy as np
import torch
import torchvision
import matplotlib.pyplot as plt
import copy
import torch.nn as nn
import torch.nn.functional as F
import os

from dataload import *
from autoencoder import *


if __name__ == "__main__":
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print(f"Launching on {device}")
    batch_size_train = 64
    batch_size_test = 64

    #train_loader, test_loader = define_mnist_loaders(batch_size_train, batch_size_test)
    train_loader, test_loader = define_landscapes_loaders(batch_size_train, batch_size_test, 
                                                          rescale=32, crop=28)

    encoding_dim = 128
    lr = 0.1
    momentum = 0.5
    log_interval = 10
    n_epochs = 5
    
    autoencoder = C_Autoencoder(28*28, encoding_dim)
    opt = torch.optim.Adam(autoencoder.parameters(), lr=lr)
    
    print(f"Began training : epochs = {n_epochs}, lr = {lr}")
    train(autoencoder, opt, trainloader=train_loader, valloader=test_loader, num_epochs=n_epochs)
    
    torch.save(autoencoder.state_dict(), "saved_models/model_c-autoenc28.sav")