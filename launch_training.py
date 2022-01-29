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
    batch_size_train = 128
    batch_size_test = 128

    #train_loader, test_loader = define_mnist_loaders(batch_size_train, batch_size_test)
    train_loader, test_loader = define_landscapes_loaders(batch_size_train, batch_size_test)

    encoding_dim = 2048
    learning_rate = 0.01
    momentum = 0.5
    log_interval = 10
    n_epochs = 20
    
    autoencoder = Autoencoder(224*224, encoding_dim)
    
    print("Began training")
    train(autoencoder, trainloader=train_loader, valloader=test_loader, num_epochs=n_epochs)
    
    torch.save(autoencoder.state_dict(), "model_parameters.sav")