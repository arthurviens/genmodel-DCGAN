import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Parameter

from src.utils import init_ortho

def l2normalize(v, eps=1e-12):
    return v / (v.norm() + eps)


class SpectralNorm(nn.Module):
    """ Spectral normalization layer
    """
    def __init__(self, module, name='weight', power_iterations=1):
        super(SpectralNorm, self).__init__()
        self.module = module
        self.name = name
        self.power_iterations = power_iterations
        if not self._made_params():
            self._make_params()

    def _update_u_v(self):
        u = getattr(self.module, self.name + "_u")
        v = getattr(self.module, self.name + "_v")
        w = getattr(self.module, self.name + "_bar")

        height = w.data.shape[0]
        for _ in range(self.power_iterations):
            v.data = l2normalize(torch.mv(torch.t(w.view(height,-1).data), u.data))
            u.data = l2normalize(torch.mv(w.view(height,-1).data, v.data))

        # sigma = torch.dot(u.data, torch.mv(w.view(height,-1).data, v.data))
        sigma = u.dot(w.view(height, -1).mv(v))
        setattr(self.module, self.name, w / sigma.expand_as(w))

    def _made_params(self):
        try:
            u = getattr(self.module, self.name + "_u")
            v = getattr(self.module, self.name + "_v")
            w = getattr(self.module, self.name + "_bar")
            return True
        except AttributeError:
            return False


    def _make_params(self):
        w = getattr(self.module, self.name)

        height = w.data.shape[0]
        width = w.view(height, -1).data.shape[1]

        u = Parameter(w.data.new(height).normal_(0, 1), requires_grad=False)
        v = Parameter(w.data.new(width).normal_(0, 1), requires_grad=False)
        u.data = l2normalize(u.data)
        v.data = l2normalize(v.data)
        w_bar = Parameter(w.data)

        del self.module._parameters[self.name]

        self.module.register_parameter(self.name + "_u", u)
        self.module.register_parameter(self.name + "_v", v)
        self.module.register_parameter(self.name + "_bar", w_bar)


    def forward(self, *args):
        self._update_u_v()
        return self.module.forward(*args)


class Self_Attn(nn.Module):
    """ Self attention Layer"""
    def __init__(self, in_dim):
        super().__init__()
        
        # Construct the conv layers
        self.query_conv = nn.Conv2d(in_channels = in_dim , out_channels = in_dim//2 , kernel_size= 1)
        self.key_conv = nn.Conv2d(in_channels = in_dim , out_channels = in_dim//2 , kernel_size= 1)
        self.value_conv = nn.Conv2d(in_channels = in_dim , out_channels = in_dim , kernel_size= 1)
        
        # Initialize gamma as 0
        self.gamma = nn.Parameter(torch.zeros(1))
        self.softmax  = nn.Softmax(dim=-1)
        
    def forward(self,x):
        """
            inputs :
                x : input feature maps( B * C * W * H)
            returns :
                out : self attention value + input feature 
                attention: B * N * N (N is Width*Height)
        """
        m_batchsize,C,width ,height = x.size()
        
        proj_query  = self.query_conv(x).view(m_batchsize, -1, width*height).permute(0,2,1) # B * N * C
        proj_key =  self.key_conv(x).view(m_batchsize, -1, width*height) # B * C * N
        energy =  torch.bmm(proj_query, proj_key) # batch matrix-matrix product
        
        attention = self.softmax(energy) # B * N * N
        proj_value = self.value_conv(x).view(m_batchsize, -1, width*height) # B * C * N
        out = torch.bmm(proj_value, attention.permute(0,2,1)) # batch matrix-matrix product
        out = out.view(m_batchsize,C,width,height) # B * C * W * H
        
        # Add attention weights onto input
        out = self.gamma*out + x
        return out, attention



##### Resnet blocks with skip connections
class ResConvBlock(nn.Module):
    """ Resnet building block that can upsample

    Args:
        in_channels : number of input channels
        out_channels : number of wanted output channels
        stride : factor of downsampling for convolution
        padding : parameter to control output size
        activation : activation function 
        last_batchnorm : True if batchnorm wanted after last layer, false otherwise
    """
    def __init__(self, in_channels, out_channels, stride=1, activation=F.leaky_relu, last_batchnorm=True):
        super(ResConvBlock, self).__init__()
        self.activation = activation
        if not isinstance(stride, int):
            raise ValueError(f"Wrong value of stride : {stride}, should be int")
        if (stride != 1) or (in_channels != out_channels):
            layers_block = [nn.Conv2d(in_channels=in_channels, out_channels=out_channels, 
                        kernel_size=1, stride=stride, bias=False)]
            if last_batchnorm:
                layers_block.append(nn.BatchNorm2d(out_channels))
            self.skip = nn.Sequential(*layers_block)
        else:
            self.skip = None
        # Both self.conv1 and self.downsample layers downsample the input when stride != 1

        layers_block = [nn.Conv2d(in_channels, in_channels, (3,3), 
                        stride=1, padding=1, bias=False),
            nn.BatchNorm2d(in_channels),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(in_channels, out_channels, (3,3), 
                        stride=stride, padding=1, bias=False)]
        if last_batchnorm:
            layers_block.append(nn.BatchNorm2d(out_channels))
        self.block = nn.Sequential(*layers_block)

    def forward(self, x):
        identity = x
        out = self.block(x)

        if self.skip is not None:
            identity = self.skip(x)

        out += identity
        out = self.activation(out)

        return out


class ResUpConvBlock(nn.Module):
    """ Resnet building block that can upsample

    Args:
        in_channels : number of input channels
        out_channels : number of wanted output channels
        stride : factor of upsampling
        padding : parameter to control output size 
        out_size : wanted output size
        activation : activation function
        last_batchnorm : True if batchnorm wanted after last layer, false otherwise
    """
    def __init__(self, in_channels, out_channels, stride=1, padding=0, out_size=None, 
                            activation=F.leaky_relu, last_batchnorm=True):
        super(ResUpConvBlock, self).__init__()
        self.activation = activation
        if not isinstance(stride, int):
            raise ValueError(f"Wrong value of stride : {stride}, should be int")
        if (stride != 1):
            if (padding != 0) and (out_size is not None):
                self.skip1 = nn.Upsample(size=out_size, mode="nearest")#, align_corners=True)
            else:
                self.skip1 = nn.Upsample(scale_factor=stride, mode="nearest")#, align_corners=True)
        if (in_channels != out_channels):
            self.skip2 = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1)

        layers_skip = []
        if hasattr(self, "skip1"):
            layers_skip.append(self.skip1)
        if hasattr(self, "skip2"):
            layers_skip.append(self.skip2)
        if len(layers_skip) > 0:
            self.skip = nn.Sequential(*layers_skip)
        else:
            self.skip = None
          
        if (stride != 1):
            if (padding != 0) and (out_size is not None):
                uplayer = nn.Upsample(size=out_size, mode="nearest")#, align_corners=True)
            else:
                uplayer = nn.Upsample(scale_factor=stride, mode="nearest")#, align_corners=True)


            layers_block = [SpectralNorm(nn.Conv2d(in_channels, in_channels, kernel_size=3,  
                stride=1, padding=1, bias=False)),
                nn.BatchNorm2d(in_channels),
                nn.ReLU(inplace=True),
                uplayer,
                SpectralNorm(nn.Conv2d(in_channels, out_channels, kernel_size=3,
                        stride=1, padding=1, bias=False))]
        else:
            layers_block = [SpectralNorm(nn.Conv2d(in_channels, in_channels, kernel_size=3,  
                stride=1, padding=1, bias=False)),
                nn.BatchNorm2d(in_channels),
                nn.ReLU(inplace=True),
                SpectralNorm(nn.Conv2d(in_channels, out_channels, kernel_size=3,
                        stride=1, padding= 1, bias=False))]

        if last_batchnorm:
            layers_block.append(nn.BatchNorm2d(out_channels))

        self.block = nn.Sequential(*layers_block)


    def forward(self, x):
        identity = x
        out = self.block(x)

        if self.skip is not None:
            identity = self.skip(x)

        out = torch.add(identity, out)
        out = self.activation(out)

        return out


################################################################################
################################################################################
################################################################################
######################### Network Architecture #################################
################################################################################
################################################################################
################################################################################

debug=False
class Discriminator(nn.Module):
    def __init__(self, attn=True):
        super().__init__()
        self.attn = attn
        ### Convolutional section
        self.block1 = ResConvBlock(3, 64, stride=1) # 64 * 128 * 128  
        self.block2 = ResConvBlock(64, 64, stride=2) # 64 * 64 * 64
        self.block3 = ResConvBlock(64, 64, stride=1) # 64 * 64 * 64
        self.block4 = ResConvBlock(64, 128, stride=2) # 128 * 32 * 32
        self.block5 = ResConvBlock(128, 128, stride=1) # 128 * 32 * 32
        self.block6 = ResConvBlock(128, 128, stride=2) # 128 * 16 * 16
        self.block7 = ResConvBlock(128, 256, stride=2) # 256 * 8 * 8
        self.block8 = ResConvBlock(256, 256, stride=1) # 256 * 8 * 8
        self.attn9 = Self_Attn(256)
        self.block10 = ResConvBlock(256, 512, stride=2) # 512 * 4 * 4
        self.block11 = ResConvBlock(512, 512, stride=1) # 512 * 4 * 4
        self.block12 = ResConvBlock(512, 1024, stride=2, last_batchnorm=False) # 1024 * 2 * 2
        

        ### Flatten layer
        self.flatten = nn.Flatten(start_dim=1)
        ### Linear section
        self.discriminator_output = nn.Sequential(
            nn.Linear(4096, 1),
            nn.Sigmoid()             #fait que la sortie est entre 0 et 1 (bien pour les probabs)
        )
        #self.discriminator_output.apply(init_ortho)
        
    def forward(self, x):
        if debug: print(f"DISCRIMINATOR {x.shape}")
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
        if self.attn:
            x, attention = self.attn9(x)
            if debug: print(f"After attn9 {x.shape}")
        x = self.block10(x)
        if debug: print(f"After block10 {x.shape}")
        x = self.block11(x)
        if debug: print(f"After block11 {x.shape}")
        x = self.block12(x)
        if debug: print(f"After block12 {x.shape}")
        x = self.flatten(x)
        if debug: print(f"After flatten {x.shape}")
        x = self.discriminator_output(x)
        if debug: print(f"After discriminator_output {x.shape}")
        return x


class Generator(nn.Module):
    def __init__(self, input_dim, attn=True):
        super().__init__()
        self.attn = attn
        self.generator_lin = nn.Sequential(
            nn.Linear(input_dim, 1024 * 2 * 2),
            nn.ReLU(True)
            )

        #self.generator_lin.apply(init_ortho)
        ### Convolutional section
        self.unflatten = nn.Unflatten(dim=1, unflattened_size=(1024, 2, 2)) # 1024 * 2 * 2
        self.block1 = ResUpConvBlock(1024, 512, stride=2) # 512 * 4 * 4 
        self.block3 = ResUpConvBlock(512, 512, stride=1) # 512 * 4 * 4
        self.block4 = ResUpConvBlock(512, 256, stride=2) # 256 * 8 * 8
        self.block5 = ResUpConvBlock(256, 256, stride=1) # 256 * 8 * 8
        self.block6 = ResUpConvBlock(256, 128, stride=2) # 128 * 16 * 16
        self.block7 = ResUpConvBlock(128, 128, stride=1) # 128 * 16 * 16
        self.block8 = ResUpConvBlock(128, 128, stride=2) # 128 * 32 * 32
        self.attn9 = Self_Attn(128)
        self.block10 = ResUpConvBlock(128, 64, stride=2) # 64 * 64 * 64
        self.block11 = ResUpConvBlock(64, 64, stride=1) # 64 * 64 * 64

        self.block12 = ResUpConvBlock(64, 3, stride=2, activation=torch.sigmoid, 
                    last_batchnorm=False) # 3 * 128 * 128

        

    def forward(self, x):
        if debug: print("GENERATOR")
        if debug: print(f"Start {x.shape}")
        x = self.generator_lin(x)
        if debug: print(f"After decoder_lin {x.shape}")
        x = self.unflatten(x)
        if debug: print(f"After unflatten {x.shape}")
        x = self.block1(x)
        if debug: print(f"After block1 {x.shape}")
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
        if self.attn:
            x, attention = self.attn9(x)
            if debug: print(f"After attn9 {x.shape}")
        x = self.block10(x)
        if debug: print(f"After block10 {x.shape}")
        x = self.block11(x)
        if debug: print(f"After block11 {x.shape}")
        x = self.block12(x)
        if debug: print(f"After block12 {x.shape}")
        return x