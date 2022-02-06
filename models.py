import torch
import torch.nn as nn
import torch.nn.functional as F
from autoencoder import ResConvBlock, ResUpConvBlock

################################################################################
################################################################################
debug=False
class Discriminator_224(nn.Module):
    def __init__(self):
        super().__init__()
        
        ### Convolutional section
        self.enc_conv1 = nn.Conv2d(3, 64, (7,7), stride=1, padding=3, bias=False) # 64 * 224 * 224
        self.batchnorm1 = nn.BatchNorm2d(64)
        self.relu1 = nn.ReLU(True)
        self.enc_conv2 = nn.Conv2d(64, 64, (7,7), stride=2, padding=3, bias=False) # 64 * 112 * 112
        self.batchnorm2 = nn.BatchNorm2d(64)
        self.relu2 = nn.ReLU(True)
        self.enc_conv3 = nn.Conv2d(64, 64, (5, 5), stride=2, padding=2, bias=False) # 64 * 56 * 56
        self.batchnorm3 = nn.BatchNorm2d(64)
        self.relu3 = nn.ReLU(True)
        self.enc_conv4 = nn.Conv2d(64, 128, (5, 5), stride=1, padding=2, bias=False) # 128 * 56 * 56
        self.batchnorm4 = nn.BatchNorm2d(128)
        self.relu4 = nn.ReLU(True)
        self.enc_conv5 = nn.Conv2d(128, 128, (3, 3), stride=2, padding=1, bias=False) # 128 * 28 * 28
        self.batchnorm5 = nn.BatchNorm2d(128)
        self.relu5 = nn.ReLU(True)
        self.enc_conv6 = nn.Conv2d(128, 256, (3, 3), stride=2, padding=1, bias=False) # 256 * 14 * 14
        self.batchnorm6 = nn.BatchNorm2d(256)
        self.relu6 = nn.ReLU(True)
        self.enc_conv7 = nn.Conv2d(256, 256, (3, 3), stride=1, padding=1, bias=False) # 256 * 14 * 14
        self.batchnorm7 = nn.BatchNorm2d(256)
        self.relu7 = nn.ReLU(True)
        self.enc_conv8 = nn.Conv2d(256, 512, (3, 3), stride=2, padding=1, bias=False) # 512 * 7 * 7
        self.batchnorm8 = nn.BatchNorm2d(512)
        self.relu8 = nn.ReLU(True)
        self.enc_conv9 = nn.Conv2d(512, 1024, (3, 3), stride=2, padding=1, bias=False) # 1024 * 4 * 4
        self.batchnorm9 = nn.BatchNorm2d(1024)
        self.relu9 = nn.ReLU(True)
        ### Flatten layer
        self.flatten = nn.Flatten(start_dim=1)
        ### Linear section
        self.discriminator_output = nn.Sequential(
            nn.Linear(1024 * 4 * 4, 1024),
            nn.ReLU(True),
            nn.Linear(1024, 1),
            nn.Sigmoid()             #fait que la sortie est entre 0 et 1 (bien pour les probabs)
        )
        
    def forward(self, x):
        x = self.enc_conv1(x)
        x = self.batchnorm1(x)
        x = self.relu1(x)
        if debug: print(f"After conv1 {x.shape}")
        x = self.enc_conv2(x)
        x = self.batchnorm2(x)
        x = self.relu2(x)
        if debug: print(f"After conv2 {x.shape}")
        x = self.enc_conv3(x)
        x = self.batchnorm3(x)
        x = self.relu3(x)
        if debug: print(f"After conv3 {x.shape}")
        x = self.enc_conv4(x)
        x = self.batchnorm4(x)
        x = self.relu4(x)
        if debug: print(f"After conv4 {x.shape}")
        x = self.enc_conv5(x)
        x = self.batchnorm5(x)
        x = self.relu5(x)
        if debug: print(f"After conv5 {x.shape}")
        x = self.enc_conv6(x)
        x = self.batchnorm6(x)
        x = self.relu6(x)
        if debug: print(f"After conv6 {x.shape}")
        x = self.enc_conv7(x)
        x = self.batchnorm7(x)
        x = self.relu7(x)
        if debug: print(f"After conv7 {x.shape}")
        x = self.enc_conv8(x)
        x = self.batchnorm8(x)
        x = self.relu8(x)
        if debug: print(f"After conv8 {x.shape}")
        x = self.enc_conv9(x)
        x = self.batchnorm9(x)
        x = self.relu9(x)
        if debug: print(f"After conv9 {x.shape}")

        x = self.flatten(x)
        if debug: print(f"After flatten {x.shape}")
        x = self.discriminator_output(x)
        if debug: print(f"After discriminator_output {x.shape}")
        return x


class Generator_224(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.generator_lin = nn.Sequential(
            nn.Linear(input_dim, 1024),
            nn.ReLU(True),
            nn.Linear(1024, 1024 * 4 * 4),
            nn.ReLU(True)
            )

        self.unflatten = nn.Unflatten(dim=1, unflattened_size=(1024, 4, 4))
        self.dec_convt1 = nn.ConvTranspose2d(1024, 512, 2, stride=2, padding=1, output_padding=1, bias=False)
        self.batchnorm1 = nn.BatchNorm2d(512)
        self.relu1 = nn.ReLU(True)
        self.dec_convt2 = nn.ConvTranspose2d(512, 256, 2, stride=2, bias=False)
        self.batchnorm2 = nn.BatchNorm2d(256)
        self.relu2 = nn.ReLU(True)
        self.dec_convt3 = nn.ConvTranspose2d(256, 256, 2, stride=1, bias=False)
        self.batchnorm3 = nn.BatchNorm2d(256)
        self.relu3 = nn.ReLU(True)
        self.dec_convt4 = nn.ConvTranspose2d(256, 128, 2, stride=2, padding=1, bias=False)
        self.batchnorm4 = nn.BatchNorm2d(128)
        self.relu4 = nn.ReLU(True)
        self.dec_convt5 = nn.ConvTranspose2d(128, 128, 2, stride=2, bias=False)
        self.batchnorm5 = nn.BatchNorm2d(128)
        self.relu5 = nn.ReLU(True)
        self.dec_convt6 = nn.ConvTranspose2d(128, 64, 2, stride=1, bias=False)
        self.batchnorm6 = nn.BatchNorm2d(64)
        self.relu6 = nn.ReLU(True)
        self.dec_convt7 = nn.ConvTranspose2d(64, 64, 2, stride=2, padding=1, bias=False)
        self.batchnorm7 = nn.BatchNorm2d(64)
        self.relu7 = nn.ReLU(True)
        self.dec_convt8 = nn.ConvTranspose2d(64, 64, 2, stride=1, bias=False)
        self.batchnorm8 = nn.BatchNorm2d(64)
        self.relu8 = nn.ReLU(True)
        self.dec_convt9 = nn.ConvTranspose2d(64, 3, 2, stride=2, padding=1, bias=False)


    def forward(self, x):
        if debug: print("GENERATOR")
        if debug: print(f"Start {x.shape}")
        x = self.generator_lin(x)
        if debug: print(f"After decoder_lin {x.shape}")
        x = self.unflatten(x)
        if debug: print(f"After unflatten {x.shape}")
        x = self.dec_convt1(x)
        x = self.batchnorm1(x)
        x = self.relu1(x)
        if debug: print(f"After transposed conv 1 {x.shape}")
        x = self.dec_convt2(x)
        x = self.batchnorm2(x)
        x = self.relu2(x)
        if debug: print(f"After transposed conv 2 {x.shape}")
        x = self.dec_convt3(x)
        x = self.batchnorm3(x)
        x = self.relu3(x)
        if debug: print(f"After transposed conv 3 {x.shape}")
        x = self.dec_convt4(x)
        x = self.batchnorm4(x)
        x = self.relu4(x)
        if debug: print(f"After transposed conv 4 {x.shape}")
        x = self.dec_convt5(x)
        x = self.batchnorm5(x)
        x = self.relu5(x)
        if debug: print(f"After transposed conv 5 {x.shape}")
        x = self.dec_convt6(x)
        x = self.batchnorm6(x)
        x = self.relu6(x)
        if debug: print(f"After transposed conv 6 {x.shape}")
        x = self.dec_convt7(x)
        x = self.batchnorm7(x)
        x = self.relu7(x)
        if debug: print(f"After transposed conv 7 {x.shape}")
        x = self.dec_convt8(x)
        x = self.batchnorm8(x)
        x = self.relu8(x)
        if debug: print(f"After transposed conv 8 {x.shape}")
        x = self.dec_convt9(x)
        if debug: print(f"After transposed conv 9 {x.shape}")
        x = torch.tanh(x)
        return x


################################################################################
################################################################################
################################################################################
################################################################################
################################################################################
################################################################################
################################################################################

class Discriminator(nn.Module):
    def __init__(self):
        super().__init__()

        ### Convolutional section
        self.enc_conv1 = nn.Conv2d(3, 64, (7,7), stride=1, padding=3, bias=False) # 64 * 224 * 224
        self.batchnorm1 = nn.BatchNorm2d(64)
        self.relu1 = nn.ReLU()
        self.block2 = ResConvBlock(64, 64, stride=2) # 64 * 112 * 112
        self.block3 = ResConvBlock(64, 128, stride=2) # 128 * 56 * 56
        self.block4 = ResConvBlock(128, 128, stride=2) # 128 * 28 * 28
        self.block5 = ResConvBlock(128, 256, stride=2) # 256 * 14 * 14
        self.block6 = ResConvBlock(256, 512, stride=2) # 512 * 7 * 7
        self.block7 = ResConvBlock(512, 1024, stride=2) # 1024 * 4 * 4

        ### Flatten layer
        self.flatten = nn.Flatten(start_dim=1)
        ### Linear section
        self.discriminator_output = nn.Sequential(
            nn.Linear(1024 * 4 * 4, 1024),
            nn.ReLU(True),
            nn.Linear(1024, 1),
            nn.Sigmoid()             #fait que la sortie est entre 0 et 1 (bien pour les probabs)
        )
        
    def forward(self, x):
        x = self.enc_conv1(x)
        if debug: print(f"After enc conv 1 {x.shape}")
        x = self.batchnorm1(x)
        x = self.relu1(x)
        x = self.block2(x)
        if debug: print(f"After block2 {x.shape}")
        x = self.block3(x)
        if debug: print(f"After block3 {x.shape}")
        x = self.block4(x)
        if debug: print(f"After block4 {x.shape}")
        x = self.block5(x)
        if debug: print(f"After block5 {x.shape}")
        x = self.block6(x)
        if debug: print(f"After block6 {x.shape}")
        x = self.block7(x)
        if debug: print(f"After block7 {x.shape}")
        x = self.flatten(x)
        if debug: print(f"After flatten {x.shape}")
        x = self.discriminator_output(x)
        if debug: print(f"After discriminator_output {x.shape}")
        return x


class Generator(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.generator_lin = nn.Sequential(
            nn.Linear(input_dim, 1024),
            nn.ReLU(True),
            nn.Linear(1024, 1024 * 4 * 4),
            nn.ReLU(True)
            )

        ### Convolutional section
        self.unflatten = nn.Unflatten(dim=1, unflattened_size=(1024, 4, 4)) # 
        self.dec_convt1 = nn.ConvTranspose2d(1024, 512, 2, stride=2, padding=1, output_padding=1)
        self.batchnorm1 = nn.BatchNorm2d(512)
        self.relu1 = nn.ReLU()
        self.block2 = ResUpConvBlock(512, 256, stride=2) # 254 * 14 * 14
        self.block3 = ResUpConvBlock(256, 128, stride=2) # 128 * 28 * 28
        self.block4 = ResUpConvBlock(128, 128, stride=2) # 128 * 56 * 56
        self.block5 = ResUpConvBlock(128, 64, stride=2) # 64 * 112 * 112
        self.block6 = ResUpConvBlock(64, 64, stride=2) # 64 * 224 * 224
        self.conv7 = nn.Conv2d(64, 3, kernel_size=(1, 1), stride=1)


    def forward(self, x):
        if debug: print("GENERATOR")
        if debug: print(f"Start {x.shape}")
        x = self.generator_lin(x)
        if debug: print(f"After decoder_lin {x.shape}")
        x = self.unflatten(x)
        if debug: print(f"After unflatten {x.shape}")
        x = self.dec_convt1(x)
        x = self.batchnorm1(x)
        x = self.relu1(x)
        if debug: print(f"After dec_convt1 {x.shape}")
        x = self.block2(x)
        if debug: print(f"After block2 {x.shape}")
        x = self.block3(x)
        if debug: print(f"After block3 {x.shape}")
        x = self.block4(x)
        if debug: print(f"After block4 {x.shape}")
        x = self.block5(x)
        if debug: print(f"After block5 {x.shape}")
        x = self.block6(x)
        if debug: print(f"After block6 {x.shape}")
        x = self.conv7(x)
        if debug: print(f"After dec_convt 7 {x.shape}")
        x = torch.tanh(x)
        return x
