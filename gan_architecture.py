import torch
import torch.nn as nn
import torch.nn.functional as F


##### Resnet blocks

class ResConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1, activation=F.leaky_relu, last_batchnorm=True):
        super(ResConvBlock, self).__init__()
        self.activation = activation
        if not isinstance(stride, int):
            raise ValueError(f"Wrong value of stride : {stride}, should be int")
        if (stride != 1) or (in_channels != out_channels):
          self.skip = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=out_channels, 
                        kernel_size=1, stride=stride, bias=False))
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


"""class ResUpConvBlock_2(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1, padding=0, out_size=None, 
                            activation=F.leaky_relu, last_batchnorm=True):
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

        layers_skip = []
        if hasattr(self, "skip1"):
            layers_skip.append(self.skip1)
        if hasattr(self, "skip2"):
            layers_skip.append(self.skip2)
        if len(layers_skip) > 0:
            self.skip = nn.Sequential(*layers_skip)
        else:
            self.skip = None
          
        layers_block = [nn.ConvTranspose2d(in_channels, in_channels, 2,  
            stride=1, bias=False),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(in_channels, out_channels, 2,
                    stride=stride, padding= 1 + padding, 
                    output_padding = padding, bias=False)]

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

        return out"""

class ResUpConvBlock(nn.Module):
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


            layers_block = [nn.Conv2d(in_channels, in_channels, kernel_size=3,  
                stride=1, padding=1, bias=False),
                nn.BatchNorm2d(in_channels),
                nn.ReLU(inplace=True),
                uplayer,
                nn.Conv2d(in_channels, out_channels, kernel_size=3,
                        stride=1, padding=1, bias=False)]
        else:
            layers_block = [nn.Conv2d(in_channels, in_channels, kernel_size=3,  
                stride=1, padding=1, bias=False),
                nn.BatchNorm2d(in_channels),
                nn.ReLU(inplace=True),
                nn.Conv2d(in_channels, out_channels, kernel_size=3,
                        stride=1, padding= 1, bias=False)]

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
################################################################################
################################################################################
################################################################################
################################################################################

debug=False
class Discriminator(nn.Module):
    def __init__(self):
        super().__init__()

        ### Convolutional section
        self.block1 = ResConvBlock(3, 64, stride=1) # 64 * 224 * 224     
        self.block2 = ResConvBlock(64, 64, stride=2) # 64 * 112 * 112
        self.block3 = ResConvBlock(64, 64, stride=1) # 64 * 112 * 112
        self.block4 = ResConvBlock(64, 128, stride=2) # 128 * 56 * 56
        #self.block4 = ResConvBlock(128, 128, stride=1) # 128 * 28 * 28
        self.block5 = ResConvBlock(128, 128, stride=2) # 128 * 28 * 28
        self.block6 = ResConvBlock(128, 256, stride=2) # 256 * 14 * 14
        self.block7 = ResConvBlock(256, 256, stride=1) # 256 * 14 * 14
        self.block8 = ResConvBlock(256, 512, stride=2) # 512 * 7 * 7
        self.block9 = ResConvBlock(512, 1024, stride=2, last_batchnorm=False) # 1024 * 4 * 4
        # self.block10 = ResConvBlock(1024, 1024, stride=2, last_batchnorm=False) # 1024 * 2 * 2

        ### Flatten layer
        self.flatten = nn.Flatten(start_dim=1)
        ### Linear section
        self.discriminator_output = nn.Sequential(
            nn.Linear(4096, 1),
            nn.Sigmoid()             #fait que la sortie est entre 0 et 1 (bien pour les probabs)
        )
        
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
        x = self.block9(x)
        if debug: print(f"After block9 {x.shape}")
        # x = self.block10(x)
        # if debug: print(f"After block10 {x.shape}")
        x = self.flatten(x)
        if debug: print(f"After flatten {x.shape}")
        x = self.discriminator_output(x)
        if debug: print(f"After discriminator_output {x.shape}")
        return x


class Generator(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.generator_lin = nn.Sequential(
            nn.Linear(input_dim, 1024 * 2 * 2),
            nn.ReLU(True)
            )

        ### Convolutional section
        self.unflatten = nn.Unflatten(dim=1, unflattened_size=(1024, 2, 2)) # 1024 * 2 * 2
        self.block1 = ResUpConvBlock(1024, 512, stride=2) # 512 * 4 * 4 
        #self.block2 = ResUpConvBlock(1024, 1024, stride=1) # 1024 * 4 * 4

        #COMMENTED OUT TO DO 128X128 IMAGES
        # self.block3 = ResUpConvBlock(512, 512, stride=2, padding=1, out_size=(7, 7)) # 512 * 7 * 7 
        #self.block4 = ResUpConvBlock(512, 512, stride=1) # 512 * 7 * 7
        self.block5 = ResUpConvBlock(512, 256, stride=2) # 256 * 14 * 14
        self.block6 = ResUpConvBlock(256, 256, stride=1) # 256 * 14 * 14
        self.block7 = ResUpConvBlock(256, 128, stride=2) # 128 * 28 * 28
        self.block8 = ResUpConvBlock(128, 128, stride=1) # 128 * 28 * 28
        self.block9 = ResUpConvBlock(128, 128, stride=2) # 128 * 56 * 56
        self.block10 = ResUpConvBlock(128, 64, stride=2) # 64 * 112 * 112
        self.block11 = ResUpConvBlock(64, 64, stride=1) # 64 * 112 * 112

        self.block12 = ResUpConvBlock(64, 3, stride=2, activation=torch.sigmoid, last_batchnorm=False) # 
        #self.block12 = ResUpConvBlock(64, 3, stride=2, activation=torch.sigmoid, last_batchnorm=False) # 


    def forward(self, x):
        if debug: print("GENERATOR")
        if debug: print(f"Start {x.shape}")
        x = self.generator_lin(x)
        if debug: print(f"After decoder_lin {x.shape}")
        x = self.unflatten(x)
        if debug: print(f"After unflatten {x.shape}")
        x = self.block1(x)
        if debug: print(f"After block1 {x.shape}")
        #x = self.block2(x)
        #if debug: print(f"After block2 {x.shape}")
        # x = self.block3(x)
        # if debug: print(f"After block3 {x.shape}")
        #x = self.block4(x)
        #if debug: print(f"After block4 {x.shape}")
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
        x = self.block12(x)
        if debug: print(f"After block12 {x.shape}")
        #x = torch.sigmoid(x)
        return x

################################################################################
################################################################################
################################################################################
################################################################################
################################################################################
################################################################################
################################################################################

def apply_weight_decay(*modules, weight_decay_factor=0., wo_bn=True):
    '''
    https://discuss.pytorch.org/t/weight-decay-in-the-optimizers-is-a-bad-idea-especially-with-batchnorm/16994/5
    Apply weight decay to pytorch model without BN;
    In pytorch:
        if group['weight_decay'] != 0:
            grad = grad.add(p, alpha=group['weight_decay'])
    p is the param;
    :param modules:
    :param weight_decay_factor:
    :return:
    '''
    for module in modules:
        for m in module.modules():
            if hasattr(m, 'weight'):
                if wo_bn and isinstance(m, torch.nn.modules.batchnorm._BatchNorm):
                    continue
                m.weight.grad += m.weight * weight_decay_factor



################################################################################
################################################################################
################################################################################
################################################################################
################################################################################
################################################################################
################################################################################
################################################################################
################################################################################
################################################################################
################################################################################
################################################################################
################################################################################
################################################################################
ch = 64

class DCDiscriminator(nn.Module):
    def __init__(self):
        super().__init__()

        ### Convolutional section
        self.block0 = ResConvBlock(3, ch, stride=1)
        self.block1 = ResConvBlock(ch, 2*ch, stride=2)
        # self.nlblock = NLBlockND(2*ch)
        self.block2 = ResConvBlock(2*ch, 4*ch, stride=2)
        self.block3 = ResConvBlock(4*ch, 8*ch, stride=2)
        self.block4 = ResConvBlock(8*ch, 16*ch, stride=2)
        self.block5 = ResConvBlock(16*ch, 16*ch, stride=1)
        self.block6 = ResConvBlock(16*ch, 16*ch, stride=2, last_batchnorm=False)
        self.act = nn.ReLU(inplace=True)
        self.pool = nn.MaxPool2d(3, stride=2)


        ## Flatten layer
        self.flatten = nn.Flatten(start_dim=1)
        ## Linear section
        self.discriminator_output = nn.Sequential(
            nn.Linear(16*ch, 1),
            nn.Sigmoid()             #fait que la sortie est entre 0 et 1 (bien pour les probabs)
        )
        
    def forward(self, x):
        # if debug: print(f"DISCRIMINATOR {x.shape}")
        x = self.block0(x)
        # if debug: print(f"After block0 {x.shape}")
        x = self.block1(x)
        # if debug: print(f"After block1 {x.shape}")
        # x = self.nlblock(x)
        # if debug: print(f"After nlblock {x.shape}")
        x = self.block2(x)
        # if debug: print(f"After block2 {x.shape}")
        x = self.block3(x)
        # if debug: print(f"After block3 {x.shape}")
        x = self.block4(x)
        # if debug: print(f"After block4 {x.shape}")
        x = self.block5(x)
        # if debug: print(f"After block5 {x.shape}")
        x = self.block6(x)
        # if debug: print(f"After block6 {x.shape}")

        x = self.act(x)
        x = self.pool(x)
        # if debug: print(f"After pool {x.shape}")

        x = self.flatten(x)
        # if debug: print(f"After flatten {x.shape}")
        x = self.discriminator_output(x)
        # if debug: print(f"After discriminator_output {x.shape}")
        return x


class DCGenerator(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.generator_lin = nn.Sequential(
            nn.Linear(input_dim, 16*ch * 4 * 4),
            nn.ReLU(True)
            )

        ### Convolutional section
        self.unflatten = nn.Unflatten(dim=1, unflattened_size=(16*ch, 4, 4))
        self.block1 = ResUpConvBlock(16*ch, 16*ch, stride=1)
        self.block2 = ResUpConvBlock(16*ch, 8*ch, stride=2)
        self.block3 = ResUpConvBlock(8*ch, 4*ch, stride=2)
        self.block4 = ResUpConvBlock(4*ch, 2*ch, stride=2)
        self.nlblock = NLBlockND(2*ch)
        self.block5 = ResUpConvBlock(2*ch, ch, stride=2)
        self.bn = nn.BatchNorm2d(ch)
        self.bnact = nn.ReLU(inplace=True)
        self.block6 = ResUpConvBlock(ch, 3, stride=2)
        self.act = nn.Tanh()


    def forward(self, x):
        x = self.generator_lin(x)
        x = self.unflatten(x)
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.block4(x)
        x = self.nlblock(x)
        x = self.block5(x)
        x = self.bn(x)
        x = self.bnact(x) 
        x = self.block6(x)

        return self.act(x)


################################################################################
################################################################################
################################################################################
################################################################################
################################################################################
################################################################################
################################################################################



#from https://github.com/tea1528/Non-Local-NN-Pytorch/blob/master/models/non_local.py
class NLBlockND(nn.Module):
    def __init__(self, in_channels, inter_channels=None, mode='embedded', 
                 dimension=2, bn_layer=True):
        """Implementation of Non-Local Block with 4 different pairwise functions but doesn't include subsampling trick
        args:
            in_channels: original channel size (1024 in the paper)
            inter_channels: channel size inside the block if not specifed reduced to half (512 in the paper)
            mode: supports Gaussian, Embedded Gaussian, Dot Product, and Concatenation
            dimension: can be 1 (temporal), 2 (spatial), 3 (spatiotemporal)
            bn_layer: whether to add batch norm
        """
        super(NLBlockND, self).__init__()

        assert dimension in [1, 2, 3]
        
        if mode not in ['gaussian', 'embedded', 'dot', 'concatenate']:
            raise ValueError('`mode` must be one of `gaussian`, `embedded`, `dot` or `concatenate`')
            
        self.mode = mode
        self.dimension = dimension

        self.in_channels = in_channels
        self.inter_channels = inter_channels

        # the channel size is reduced to half inside the block
        if self.inter_channels is None:
            self.inter_channels = in_channels // 2
            if self.inter_channels == 0:
                self.inter_channels = 1
        
        # assign appropriate convolutional, max pool, and batch norm layers for different dimensions
        if dimension == 3:
            conv_nd = nn.Conv3d
            max_pool_layer = nn.MaxPool3d(kernel_size=(1, 2, 2))
            bn = nn.BatchNorm3d
        elif dimension == 2:
            conv_nd = nn.Conv2d
            max_pool_layer = nn.MaxPool2d(kernel_size=(2, 2))
            bn = nn.BatchNorm2d
        else:
            conv_nd = nn.Conv1d
            max_pool_layer = nn.MaxPool1d(kernel_size=(2))
            bn = nn.BatchNorm1d

        # function g in the paper which goes through conv. with kernel size 1
        self.g = conv_nd(in_channels=self.in_channels, out_channels=self.inter_channels, kernel_size=1)

        # add BatchNorm layer after the last conv layer
        if bn_layer:
            self.W_z = nn.Sequential(
                    conv_nd(in_channels=self.inter_channels, out_channels=self.in_channels, kernel_size=1),
                    bn(self.in_channels)
                )
            # from section 4.1 of the paper, initializing params of BN ensures that the initial state of non-local block is identity mapping
            nn.init.constant_(self.W_z[1].weight, 0)
            nn.init.constant_(self.W_z[1].bias, 0)
        else:
            self.W_z = conv_nd(in_channels=self.inter_channels, out_channels=self.in_channels, kernel_size=1)

            # from section 3.3 of the paper by initializing Wz to 0, this block can be inserted to any existing architecture
            nn.init.constant_(self.W_z.weight, 0)
            nn.init.constant_(self.W_z.bias, 0)

        # define theta and phi for all operations except gaussian
        if self.mode == "embedded" or self.mode == "dot" or self.mode == "concatenate":
            self.theta = conv_nd(in_channels=self.in_channels, out_channels=self.inter_channels, kernel_size=1)
            self.phi = conv_nd(in_channels=self.in_channels, out_channels=self.inter_channels, kernel_size=1)
        
        if self.mode == "concatenate":
            self.W_f = nn.Sequential(
                    nn.Conv2d(in_channels=self.inter_channels * 2, out_channels=1, kernel_size=1),
                    nn.ReLU()
                )
            
    def forward(self, x):
        """
        args
            x: (N, C, T, H, W) for dimension=3; (N, C, H, W) for dimension 2; (N, C, T) for dimension 1
        """

        batch_size = x.size(0)
        
        # (N, C, THW)
        # this reshaping and permutation is from the spacetime_nonlocal function in the original Caffe2 implementation
        g_x = self.g(x).view(batch_size, self.inter_channels, -1)
        g_x = g_x.permute(0, 2, 1)

        if self.mode == "gaussian":
            theta_x = x.view(batch_size, self.in_channels, -1)
            phi_x = x.view(batch_size, self.in_channels, -1)
            theta_x = theta_x.permute(0, 2, 1)
            f = torch.matmul(theta_x, phi_x)

        elif self.mode == "embedded" or self.mode == "dot":
            theta_x = self.theta(x).view(batch_size, self.inter_channels, -1)
            phi_x = self.phi(x).view(batch_size, self.inter_channels, -1)
            theta_x = theta_x.permute(0, 2, 1)
            f = torch.matmul(theta_x, phi_x)

        elif self.mode == "concatenate":
            theta_x = self.theta(x).view(batch_size, self.inter_channels, -1, 1)
            phi_x = self.phi(x).view(batch_size, self.inter_channels, 1, -1)
            
            h = theta_x.size(2)
            w = phi_x.size(3)
            theta_x = theta_x.repeat(1, 1, 1, w)
            phi_x = phi_x.repeat(1, 1, h, 1)
            
            concat = torch.cat([theta_x, phi_x], dim=1)
            f = self.W_f(concat)
            f = f.view(f.size(0), f.size(2), f.size(3))
        
        if self.mode == "gaussian" or self.mode == "embedded":
            f_div_C = F.softmax(f, dim=-1)
        elif self.mode == "dot" or self.mode == "concatenate":
            N = f.size(-1) # number of position in x
            f_div_C = f / N
        
        y = torch.matmul(f_div_C, g_x)
        
        # contiguous here just allocates contiguous chunk of memory
        y = y.permute(0, 2, 1).contiguous()
        y = y.view(batch_size, self.inter_channels, *x.size()[2:])
        
        W_y = self.W_z(y)
        # residual connection
        z = W_y + x

        return z
