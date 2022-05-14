import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision

import pytorch_lightning as pl
from collections import OrderedDict

from src.gan_architecture import Generator, Discriminator
from src.dataload import Rescale, RandomCrop, ToTensor, Grayscale, CustomDataset

import torch
from torch.utils.data import DataLoader
from torch import is_tensor, from_numpy
from torchvision import transforms
from sklearn.model_selection import train_test_split
import os



class GANDataModule(pl.LightningDataModule):
    def __init__(
        self,
        data_dir: str = 'data/lhq256',
        batch_size: int = 64,
        num_workers: int = 8,
        rescale: int = 140,
        crop: int = 128,
        rgb: bool = True,
    ):
        super().__init__()
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.rescale = rescale 
        self.crop = crop
        self.rgb = rgb

        if rgb:
            self.transform = transforms.Compose([
                                               Rescale(rescale),
                                               RandomCrop(crop),
                                               ToTensor(),
                                               #PyTMinMaxScalerVectorized(),
                                               transforms.RandomHorizontalFlip(p=0.5)
                                           ])
        else:
            self.transform = transforms.Compose([
                                               Rescale(rescale),
                                               RandomCrop(crop),
                                               Grayscale(),
                                               ToTensor(),
                                               #PyTMinMaxScalerVectorized(),
                                               transforms.RandomHorizontalFlip(p=0.5)
                                           ])
        # self.dims is returned when you call dm.size()
        # Setting default dims here because we know them.
        # Could optionally be assigned dynamically in dm.setup()
        self._dims = (3, crop, crop)

    def prepare_data(self):
        # download
        ls = os.listdir(self.data_dir)
        self.train_files, self.val_files = train_test_split(ls, test_size=0.05, 
                                                shuffle=True, random_state=113)
        self.train_set = CustomDataset(self.data_dir, 
            self.train_files, 
            transform = self.transform) 
        self.val_set = CustomDataset(self.data_dir,
            self.val_files, 
            transform = self.transform) 
        

    def setup(self, stage=None):
        """
        # Assign train/val datasets for use in dataloaders
        if stage == "fit" or stage is None:
            mnist_full = MNIST(self.data_dir, train=True, transform=self.transform)
            self.mnist_train, self.mnist_val = random_split(mnist_full, [55000, 5000])

        # Assign test dataset for use in dataloader(s)
        if stage == "test" or stage is None:
            self.mnist_test = MNIST(self.data_dir, train=False, transform=self.transform)
        """
        self.prepare_data()

    def train_dataloader(self):
        return DataLoader(
            self.train_set,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=True
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_set, 
            batch_size=self.batch_size, 
            num_workers=self.num_workers,
            shuffle=False
        )


class GAN(pl.LightningModule):
    def __init__(
        self,
        channels,
        width,
        height,
        latent_dim: int = 128,
        lrG: float = 0.00001,
        lrD: float = 0.00004,
        b1: float = 0.5,
        b2: float = 0.999,
        batch_size: int = 128,
        **kwargs
    ):
        super().__init__()
        self.save_hyperparameters()

        self.latent_dim = latent_dim
        self.lrG = lrG 
        self.lrD = lrD
        self.b1 = b1
        self.b2 = b2
        self.batch_size = batch_size
        
        self._data_shape = (channels, width, height)
        self.generator = Generator(latent_dim, attn=False)
        self.discriminator = Discriminator(attn=False)

        self.validation_z = torch.randn(8, latent_dim)

        self.example_input_array = torch.zeros(2, latent_dim)

    def forward(self, z):
        return self.generator(z)

    def adversarial_loss(self, y_hat, y):
        return F.binary_cross_entropy(y_hat, y)

    def training_step(self, batch, batch_idx, optimizer_idx):
        # sample noise
        z = torch.randn(batch.shape[0], self.latent_dim)
        z = z.type_as(batch)

        # train generator
        if optimizer_idx == 0:

            # generate images
            self.generated_imgs = self(z)

            # ground truth result (ie: all fake)
            # put on GPU because we created this tensor inside training_loop
            valid = torch.ones(batch.size(0), 1)
            valid = valid.type_as(batch)

            # adversarial loss is binary cross-entropy
            g_loss = self.adversarial_loss(self.discriminator(self.generated_imgs), valid)
            self.log("g_loss/train", g_loss)
            tqdm_dict = {"g_loss": g_loss}
            output = OrderedDict({"loss": g_loss, "progress_bar": tqdm_dict, "log": tqdm_dict})
            return output

        # train discriminator
        if optimizer_idx == 1:
            # Measure discriminator's ability to classify real from generated samples

            # how well can it label as real?
            valid = torch.ones(batch.size(0), 1)
            valid = valid.type_as(batch)

            real_loss = self.adversarial_loss(self.discriminator(batch), valid)

            # how well can it label as fake?
            fake = torch.zeros(batch.size(0), 1)
            fake = fake.type_as(batch)

            fake_loss = self.adversarial_loss(self.discriminator(self(z).detach()), fake)

            # discriminator loss is the average of these
            d_loss = (real_loss + fake_loss) / 2
            self.log("dloss/train", d_loss)
            tqdm_dict = {"d_loss": d_loss}
            output = OrderedDict({"loss": d_loss, "progress_bar": tqdm_dict, "log": tqdm_dict})
            return output

    def configure_optimizers(self):
        lrG = self.lrG
        lrD = self.lrD
        b1 = self.b1
        b2 = self.b2

        opt_g = torch.optim.Adam(self.generator.parameters(), lr=lrG, betas=(b1, b2))
        opt_d = torch.optim.Adam(self.discriminator.parameters(), lr=lrD, betas=(b1, b2))
        return [opt_g, opt_d], []

    def train_epoch_end(self):
        z = self.validation_z.type_as(self.generator.model[0].weight)

        # log sampled images
        sample_imgs = self(z)
        grid = torchvision.utils.make_grid(sample_imgs)
        self.logger.experiment.add_image("generated_images", grid, self.current_epoch)