import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision.utils import save_image, make_grid
from src.dataload import *
from src.gan_architecture import *
from tqdm import tqdm
import pandas as pd
import numpy as np
from src.utils import get_n_params, accuracy, write_params
from src.utils import apply_weight_decay, get_epoch_from_log
import argparse 

from pytorch_lightning import Trainer
from pytorch_lightning.loggers import TensorBoardLogger
from src.lightning_model import GAN, GANDataModule

#device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

## Data set parameters
ds = "data/lhq_256"
bs = 256
rescale_size = 140
crop_size = 128

archi_info = "upsample type : nearest"

#Optimizer parameters
lrG = 0.00001
lrD = 0.00004
beta1 = 0.5
beta2 = 0.999
weight_decayG = 0
weight_decayD = 0

#Input of generator
z_dim = 128

#Training parameters
# savefile = 'res-gan-bilinear'
n_epoch = 500
#save_frequency = 1
#k = 2 #Facteur d'apprentissage discriminateur
#n_generated_save = 9 #number of images to output at each save_frequency epochs

"""if --midsave args is passed is activated, save the
evolution models every n_midsave epochs"""
#n_midsave = 100

#Labels for discriminator for fake and real images
#label_reals = 0.9 
#label_fakes = 0.0
#labels = torch.full((bs, 1), label_reals, dtype=torch.float, device=device)


data = GANDataModule(data_dir=ds, batch_size=bs, 
                    rescale=rescale_size, crop=crop_size)
s = data._dims
dnn = GAN(s[0], s[1], s[2], latent_dim=z_dim,
    lrG=lrG, lrD=lrD, b1=beta1, b2=beta2, batch_size=bs)

logger = TensorBoardLogger("lightning_logs", name="gan")


trainer = Trainer(
    gpus=[0, 1],
    accelerator='dp',
    max_epochs=n_epoch,
    logger=logger
)
trainer.fit(dnn, data)