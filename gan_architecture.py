import torch
import torch.nn as nn
import torch.nn.functional as F


##### Resnet blocks

class ResConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1, activation=F.leaky_relu):
        super(ResConvBlock, self).__init__()
        self.activation = activation
        if not isinstance(stride, int):
            raise ValueError(f"Wrong value of stride : {stride}, should be int")
        if (stride != 1) or (in_channels != out_channels):
          self.skip = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=out_channels, 
                        kernel_size=1, stride=stride, bias=False),
            nn.BatchNorm2d(out_channels))
        else:
          self.skip = None
        # Both self.conv1 and self.downsample layers downsample the input when stride != 1

        self.block = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, (3,3), 
                        stride=1, padding=1, bias=False),
            nn.BatchNorm2d(in_channels),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(in_channels, out_channels, (3,3), 
                        stride=stride, padding=1, bias=False),
            nn.BatchNorm2d(out_channels)
        )

    def forward(self, x):
        identity = x
        out = self.block(x)

        if self.skip is not None:
            identity = self.skip(x)

        out += identity
        out = self.activation(out)

        return out


class ResUpConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1, padding=0, out_size=None, activation=F.relu):
        super(ResUpConvBlock, self).__init__()
        self.activation = activation
        if not isinstance(stride, int):
            raise ValueError(f"Wrong value of stride : {stride}, should be int")
        if (stride != 1):
            if (padding != 0) and (out_size is not None):
                self.skip1 = nn.Upsample(size=out_size, mode="bilinear", align_corners=True)
            else:
                self.skip1 = nn.Upsample(scale_factor=stride, mode="bilinear", align_corners=True)
        if (in_channels != out_channels):
            self.skip2 = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1)
        if (hasattr(self, "skip1")) and (hasattr(self, "skip2")):
            self.skip = nn.Sequential(
                self.skip1,
                self.skip2,
                nn.BatchNorm2d(out_channels)
            )
        elif (hasattr(self, "skip1")) and (not hasattr(self, "skip2")):
            self.skip = nn.Sequential(
                self.skip1,
                nn.BatchNorm2d(out_channels)
            )
        elif (not hasattr(self, "skip1")) and (hasattr(self, "skip2")):
            self.skip = nn.Sequential(
                self.skip2,
                nn.BatchNorm2d(out_channels)
            )
        else:
          self.skip = None
          

        self.block = nn.Sequential(
            nn.ConvTranspose2d(in_channels, in_channels, 2,  
                        stride=1, bias=False),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(in_channels, out_channels, 2,
                    stride=stride, padding= 1 + padding, 
                    output_padding = padding, bias=False), 
            nn.BatchNorm2d(out_channels)
        )

    def forward(self, x):
        identity = x
        out = self.block(x)

        if self.skip is not None:
            identity = self.skip(x)

        out = torch.add(identity, out)
        # out = F.relu(out)
        out = self.activation(out)

        return out


################################################################################
################################################################################
################################################################################
################################################################################
################################################################################
################################################################################
################################################################################


debug=False
class Discriminator(nn.Module):
    def __init__(self):
        super().__init__()

        ### Convolutional section
        self.block1 = ResConvBlock(3, 64, stride=1)
        self.block2 = ResConvBlock(64, 64, stride=2) # 64 * 112 * 112
        self.block3 = ResConvBlock(64, 128, stride=2) # 128 * 56 * 56
        self.block4 = ResConvBlock(128, 128, stride=2) # 128 * 28 * 28
        self.block5 = ResConvBlock(128, 256, stride=2) # 256 * 14 * 14
        self.block6 = ResConvBlock(256, 256, stride=1) # 256 * 14 * 14
        self.block7 = ResConvBlock(256, 512, stride=2) # 512 * 7 * 7
        self.block8 = ResConvBlock(512, 512, stride=1) # 512 * 7 * 7
        self.block9 = ResConvBlock(512, 1024, stride=2) # 1024 * 4 * 4
        self.block10 = ResConvBlock(1024, 2048, stride=4) # 2048 * 1 * 1

        ### Flatten layer
        self.flatten = nn.Flatten(start_dim=1)
        ### Linear section
        self.discriminator_output = nn.Sequential(
            nn.Linear(2048, 1),
            nn.Sigmoid()             #fait que la sortie est entre 0 et 1 (bien pour les probabs)
        )
        
    def forward(self, x):
        x = self.block1(x)
        if debug: print(f"After block1 {x.shape}")
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
        x = self.block8(x)
        if debug: print(f"After block8 {x.shape}")
        x = self.block9(x)
        if debug: print(f"After block9 {x.shape}")
        x = self.block10(x)
        if debug: print(f"After block10 {x.shape}")
        x = self.flatten(x)
        if debug: print(f"After flatten {x.shape}")
        x = self.discriminator_output(x)
        if debug: print(f"After discriminator_output {x.shape}")
        return x


class Generator(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.generator_lin = nn.Sequential(
            nn.Linear(input_dim, 2048 * 2 * 2),
            nn.ReLU(True)
            )

        ### Convolutional section
        self.unflatten = nn.Unflatten(dim=1, unflattened_size=(2048, 2, 2)) # 
        self.block1 = ResUpConvBlock(2048, 1024, stride=2) # 
        self.block2 = ResUpConvBlock(1024, 512, stride=2, padding=1, out_size=(7, 7)) # 512 * 7 * 7 
        self.block3 = ResUpConvBlock(512, 512, stride=1) # 512 * 7 * 7
        self.block4 = ResUpConvBlock(512, 256, stride=2) # 256 * 14 * 14
        self.block5 = ResUpConvBlock(256, 256, stride=1) # 256 * 14 * 14
        self.block6 = ResUpConvBlock(256, 128, stride=2) # 128 * 28 * 28
        self.block7 = ResUpConvBlock(128, 128, stride=1) # 128 * 28 * 28
        self.block8 = ResUpConvBlock(128, 128, stride=2) # 
        self.block9 = ResUpConvBlock(128, 64, stride=2) # 
        self.block10 = ResUpConvBlock(64, 64, stride=1) # 
        self.block11 = ResUpConvBlock(64, 3, stride=2, activation=torch.sigmoid) # 


    def forward(self, x):
        if debug: print("GENERATOR")
        if debug: print(f"Start {x.shape}")
        x = self.generator_lin(x)
        if debug: print(f"After decoder_lin {x.shape}")
        x = self.unflatten(x)
        if debug: print(f"After unflatten {x.shape}")
        x = self.block1(x)
        if debug: print(f"After block1 {x.shape}")
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
        x = self.block8(x)
        if debug: print(f"After block8 {x.shape}")
        x = self.block9(x)
        if debug: print(f"After block9 {x.shape}")
        x = self.block10(x)
        if debug: print(f"After block10 {x.shape}")
        x = self.block11(x)
        if debug: print(f"After block11 {x.shape}")
        #x = torch.sigmoid(x)
        return x
