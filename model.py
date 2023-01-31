import os
import numpy as np


import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn as nn
import torchvision
import torchvision.utils as vutils
import random # to generate fake labels
import pandas as pd
import torchvision.transforms as T
from torch.utils.data import Dataset, DataLoader

import os
import numpy as np
from PIL import Image

def set_seed(seed=42):
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    # When running on the CuDNN backend, two further options must be set
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    # Set a fixed value for the hash seed
    os.environ["PYTHONHASHSEED"] = str(seed)
    print(f"Random seed set as {seed}")


class generator(nn.Module):
    # initializers
    def __init__(self, z = 100, ch = 64):
        super().__init__()
        
        self.deconv1 = nn.ConvTranspose2d(z, ch*8, 2, 2)
        self.deconv2 = nn.ConvTranspose2d(ch*8, ch*4, 4, 2, 1)
        self.deconv2_bn = nn.BatchNorm2d(ch*4)
        self.deconv3 = nn.ConvTranspose2d(ch*4, ch*2, 4, 2, 1)
        self.deconv3_bn = nn.BatchNorm2d(ch*2)
        self.deconv4 = nn.ConvTranspose2d(ch*2, ch, 4, 2, 1)
        self.deconv4_bn = nn.BatchNorm2d(ch)
        self.deconv5 = nn.ConvTranspose2d(ch, 3, 4, 2, 1)

    # forward method
    def forward(self, x):
        x = F.leaky_relu(self.deconv1(x))
        x = F.leaky_relu(self.deconv2_bn(self.deconv2(x)))
        x = F.leaky_relu(self.deconv3_bn(self.deconv3(x)))
        x = F.leaky_relu(self.deconv4_bn(self.deconv4(x)))
        x = torch.tanh(self.deconv5(x))

        return x
    
class discriminator(nn.Module):
    # initializers
    def __init__(self, ch = 16):
        super(discriminator, self).__init__()
        self.conv1 = nn.Conv2d(3, ch, 4, 2, 1)
        self.conv1_bn = nn.InstanceNorm2d(ch, affine=True) # affine to learn affine parameters
        
        self.conv2 = nn.Conv2d(ch, ch*2, 4, 2, 1)
        self.conv2_bn = nn.InstanceNorm2d(ch*2, affine=True) # affine to learn affine parameters
        self.conv3 = nn.Conv2d(ch*2, ch*4, 4, 2, 1)
        self.conv3_bn = nn.InstanceNorm2d(ch*4, affine=True) # affine to learn affine parameters
        self.conv4 = nn.Conv2d(ch*4, ch*8, 4, 2, 1)
        self.conv4_bn = nn.InstanceNorm2d(ch*8, affine=True) # affine to learn affine parameters
        self.conv5 = nn.Conv2d(ch*8, 1, 2, 2)
        self.conv5_bn = nn.InstanceNorm2d(1, affine=True) # affine to learn affine parameters

        self.avgpool = nn.AdaptiveAvgPool2d((1,1))
    # forward method
    def forward(self, x):
        x = F.leaky_relu(self.conv1(x), 0.2)
        x = F.leaky_relu((self.conv2(x)), 0.2)
        x = F.leaky_relu((self.conv3(x)), 0.2)
        x = F.leaky_relu(self.conv4_bn(self.conv4(x)), 0.2)
        x = self.conv5(x)
        
        x = self.avgpool(x)
        
        return x

class AdaIN(nn.Module):
    def __init__(self):
        super().__init__()

    def mu(self, x):
        """ Takes a (n,c,h,w) tensor as input and returns the average across
        it's spatial dimensions as (h,w) tensor [See eq. 5 of paper]"""
        return torch.sum(x,(2,3))/(x.shape[2]*x.shape[3])

    def sigma(self, x):
        """ Takes a (n,c,h,w) tensor as input and returns the standard deviation
        across it's spatial dimensions as (h,w) tensor [See eq. 6 of paper] Note
        the permutations are required for broadcasting"""
        return torch.sqrt((torch.sum((x.permute([2,3,0,1])-self.mu(x)).permute([2,3,0,1])**2,(2,3))+0.000000023)/(x.shape[2]*x.shape[3]))

    def forward(self, x, y):
        """ Takes a content embeding x and a style embeding y and changes
        transforms the mean and standard deviation of the content embedding to
        that of the style. [See eq. 8 of paper] Note the permutations are
        required for broadcasting"""
        return (self.sigma(y)*((x.permute([2,3,0,1])-self.mu(x))/self.sigma(x)) + self.mu(y)).permute([2,3,0,1])

class generator_style(nn.Module):
    # initializers
    def __init__(self, z=512, ch=64, img_size=32, gamma=0.01, device=None):
        super().__init__()
        #z+=1 # Adding the class dimension to each random vector
        
        torch.cuda.device(device)
        self.AdaIN = AdaIN() # just an operatioin

        self.constants = nn.Parameter(torch.randn((1, ch, 4, 4)))
        self.device= device
        self.sv = nn.Sequential(
            nn.Linear(z,z), nn.LeakyReLU(0.2),
            nn.Linear(z,z), nn.LeakyReLU(0.2),
            nn.Linear(z,z), nn.LeakyReLU(0.2),
            nn.Linear(z,z), nn.LeakyReLU(0.2),
            nn.Linear(z,z), nn.LeakyReLU(0.2),
            nn.Linear(z,z), nn.LeakyReLU(0.2),
            nn.Linear(z,z), nn.LeakyReLU(0.2),
            nn.Linear(z,z), nn.LeakyReLU(0.2),
        )

#         self.A11 = nn.Linear(z, ch)
#         self.A12 = nn.Linear(z, ch)
#         self.A21 = nn.Linear(z, ch)
#         self.A22 = nn.Linear(z, ch)
#         self.A31 = nn.Linear(z, ch)
#         self.A32 = nn.Linear(z, ch)
#         self.A41 = nn.Linear(z, ch)
#         self.A42 = nn.Linear(z, ch)

        self.A11 = nn.Linear(z, z)
        self.A12 = nn.Linear(z, z)
        self.A21 = nn.Linear(z, z)
        self.A22 = nn.Linear(z, z)
        self.A31 = nn.Linear(z, z)
        self.A32 = nn.Linear(z, z)
        self.A41 = nn.Linear(z, z)
        self.A42 = nn.Linear(z, z)
        
        # Noise learnt scalling factors for each latent dim of noise
        self.B11 = nn.Parameter(torch.randn((1, 1, 4, 4), device=device))
        self.B12 = nn.Parameter(torch.randn((1, 1, 4, 4), device=device))
        self.B21 = nn.Parameter(torch.randn((1, 1, 8, 8), device=device))
        self.B22 = nn.Parameter(torch.randn((1, 1, 8, 8), device=device))
        self.B31 = nn.Parameter(torch.randn((1, 1, 16, 16), device=device))
        self.B32 = nn.Parameter(torch.randn((1, 1, 16, 16), device=device))
        self.B41 = nn.Parameter(torch.randn((1, 1, 32, 32), device=device))
        self.B42 = nn.Parameter(torch.randn((1, 1, 32, 32), device=device))

        self.up = nn.Upsample(scale_factor=2, align_corners=True, mode='bilinear')
        self.conv41 = nn.Conv2d(z, ch, 3, 1, 1)
        self.conv81 = nn.Conv2d(z, ch, 3, 1, 1)
        self.conv82 = nn.Conv2d(z, ch, 3, 1, 1)
        self.conv161 = nn.Conv2d(z, ch, 3, 1, 1)
        self.conv162 = nn.Conv2d(z, ch, 3, 1, 1)
        self.conv321 = nn.Conv2d(z, ch, 3, 1, 1)
        self.conv322 = nn.Conv2d(z, ch, 3, 1, 1)
        
        self.toRGB = nn.Conv2d(z, 3, 1, 1)

        self.tanh = nn.Tanh()

    # forward method
    def forward(self, x):
        # x is noise vector
        B = x.shape[0]
        # Style Vectorizer
        w = self.sv(x)

        y11 = self.A11(w)
        y12 = self.A12(w)
        y21 = self.A21(w)
        y22 = self.A22(w)
        y31 = self.A31(w)
        y32 = self.A32(w)
        y41 = self.A41(w)
        y42 = self.A42(w)

        # To-Do: styles Mixing Regularisation
        # To-Do: weights demodulation to remove bubble artifacts

        # Synthesis Network
        # 4 x 4 size
        c = self.constants + self.B11*torch.randn((B, 1, 4, 4), device=self.device)
        x = self.AdaIN(c, y11)
        x = self.conv41(x)
        x = x + self.B12*torch.randn((B, 1, 4, 4), device=self.device)
        x = F.leaky_relu(x, 0.2)
        x = self.AdaIN(x, y12 )
        #print(x.shape)
        # 8 x 8 size
        x = self.up(x)
        x = self.conv81(x)
        x = x + self.B21*torch.randn((B, 1, 8, 8), device=self.device)
        x = F.leaky_relu(x, 0.2)
        x = self.AdaIN(x, y21)
        x = self.conv82(x)
        x = x + self.B22*torch.randn((B, 1, 8, 8), device=self.device)
        x = F.leaky_relu(x, 0.2)
        x = self.AdaIN(x, y22)
        #print(x.shape)

        # 16 x 16 size
        x = self.up(x)
        x = self.conv161(x)
        x = x + self.B31*torch.randn((B, 1, 16, 16), device=self.device)
        x = F.leaky_relu(x, 0.2)
        x = self.AdaIN(x, y31)
        x = self.conv162(x)
        x = x + self.B32*torch.randn((B, 1, 16, 16), device=self.device)
        x = F.leaky_relu(x, 0.2)
        x = self.AdaIN(x, y32)
        #print(x.shape)

        # 32 x 32 size
        x = self.conv321(x)
        x = self.up(x)
        x = x + self.B41*torch.randn((B, 1, 32, 32), device=self.device)
        x = F.leaky_relu(x, 0.2)
        x = self.AdaIN(x, y41)
        x = self.conv322(x)
        x = x + self.B42*torch.randn((B, 1, 32, 32), device=self.device)
        x = F.leaky_relu(x, 0.2)
        x = self.AdaIN(x, y42)

        #x = F.leaky_relu(x, 0.2)
        x = self.toRGB(x)
        x = self.tanh(x)
        #x = F.leaky_relu(x, 0.2)

        return x

class g_test(nn.Module):
    # initializers
    def __init__(self, z=512, ch=64, img_size=32, gamma=0.01, device='cuda'):
        super().__init__()
        #z+=1 # Adding the class dimension to each random vector
        
        self.AdaIN = AdaIN() # just an operatioin

        self.constants = nn.Parameter(torch.randn((1, ch, 4, 4)))

        self.sv = nn.Sequential(
            nn.Linear(z,z), nn.LeakyReLU(0.2),
            nn.Linear(z,z), nn.LeakyReLU(0.2),
            nn.Linear(z,z), nn.LeakyReLU(0.2),
            nn.Linear(z,z), nn.LeakyReLU(0.2),
            nn.Linear(z,z), nn.LeakyReLU(0.2),
            nn.Linear(z,z), nn.LeakyReLU(0.2),
            nn.Linear(z,z), nn.LeakyReLU(0.2),
            nn.Linear(z,z), nn.LeakyReLU(0.2),
        )

        self.A11 = nn.Linear(z, ch)
        self.A12 = nn.Linear(z, ch)
        self.A21 = nn.Linear(z, ch)
        self.A22 = nn.Linear(z, ch)
        self.A31 = nn.Linear(z, ch)
        self.A32 = nn.Linear(z, ch)
        self.A41 = nn.Linear(z, ch)
        self.A42 = nn.Linear(z, ch)

        # Noise learnt scalling factors for each latent dim of noise
        self.B11 = nn.Parameter(torch.randn((1, 1, 4, 4)))
        self.B12 = nn.Parameter(torch.randn((1, 1, 4, 4)))
        self.B21 = nn.Parameter(torch.randn((1, 1, 8, 8)))
        self.B22 = nn.Parameter(torch.randn((1, 1, 8, 8)))
        self.B31 = nn.Parameter(torch.randn((1, 1, 16, 16)))
        self.B32 = nn.Parameter(torch.randn((1, 1, 16, 16)))
        self.B41 = nn.Parameter(torch.randn((1, 1, 32, 32)))
        self.B42 = nn.Parameter(torch.randn((1, 1, 32, 32)))

        self.up = nn.Upsample(scale_factor=2, align_corners=True, mode='bilinear')
        self.conv41 = nn.Conv2d(ch, ch, 3, 1, 1)
        self.conv81 = nn.Conv2d(ch, ch, 3, 1, 1)
        self.conv82 = nn.Conv2d(ch, ch, 3, 1, 1)
        self.conv161 = nn.Conv2d(ch, ch, 3, 1, 1)
        self.conv162 = nn.Conv2d(ch, ch, 3, 1, 1)
        self.conv321 = nn.Conv2d(ch, ch, 3, 1, 1)
        self.conv322 = nn.Conv2d(ch, ch, 3, 1, 1)
        
        self.toRGB = nn.Conv2d(ch, 3, 1, 1)

        self.tanh = nn.Tanh()

    # forward method
    def forward(self, x):
        # x is noise vector
        B = x.shape[0]
        # Style Vectorizer
        w = self.sv(x)

        y11 = self.A11(w)
        y12 = self.A12(w)
        y21 = self.A21(w)
        y22 = self.A22(w)
        y31 = self.A31(w)
        y32 = self.A32(w)
        y41 = self.A41(w)
        y42 = self.A42(w)

        # To-Do: styles Mixing Regularisation
        # To-Do: weights demodulation to remove bubble artifacts

        # Synthesis Network
        # 4 x 4 size
               
        c = self.constants + self.B11*torch.randn((B, 1, 4, 4), device=device)
        x = self.AdaIN(c, y11)
        x = self.conv41(x)
        x = x + self.B12*torch.randn((B, 1, 4, 4), device=device)
        x = F.leaky_relu(x, 0.2)
        x = self.AdaIN(x, y12 )

        # 8 x 8 size
        x = self.up(x)
        x = self.conv81(x)
        x = x + self.B21*torch.randn((B, 1, 8, 8), device=device)
        x = F.leaky_relu(x, 0.2)
        x = self.AdaIN(x, y21)
        x = self.conv82(x)
        x = x + self.B22*torch.randn((B, 1, 8, 8), device=device)
        x = F.leaky_relu(x, 0.2)
        x = self.AdaIN(x, y22)

        # 16 x 16 size
        x = self.up(x)
        x = self.conv161(x)
        x = x + self.B31*torch.randn((B, 1, 16, 16), device=device)
        x = F.leaky_relu(x, 0.2)
        x = self.AdaIN(x, y31)
        x = self.conv162(x)
        x = x + self.B32*torch.randn((B, 1, 16, 16), device=device)
        x = F.leaky_relu(x, 0.2)
        x = self.AdaIN(x, y32)
        #print(x.shape)

        # 32 x 32 size
        x = self.conv321(x)
        x = self.up(x)
        x = x + self.B41*torch.randn((B, 1, 32, 32), device=device)
        x = F.leaky_relu(x, 0.2)
        x = self.AdaIN(x, y41)
        x = self.conv322(x)
        x = x + self.B42*torch.randn((B, 1, 32, 32), device=device)
        x = F.leaky_relu(x, 0.2)
        x = self.AdaIN(x, y42)

        x = self.toRGB(x)
        x = self.tanh(x)

        return x

class g_stylegan2(nn.Module):
    # initializers
    def __init__(self, z=512, ch=64, img_size=32, gamma=0.01, device='cuda'):
        super().__init__()
        #z+=1 # Adding the class dimension to each random vector
        
        self.AdaIN = AdaIN() # just an operatioin

        self.constants = nn.Parameter(torch.randn((1, ch, 4, 4)))
        self.constants_b = nn.Parameter(torch.randn((1, ch, 1, 1)))
        self.device = device
        
        self.sv = nn.Sequential(
            nn.Linear(z,z), nn.LeakyReLU(0.2),
            nn.Linear(z,z), nn.LeakyReLU(0.2),
            nn.Linear(z,z), nn.LeakyReLU(0.2),
            nn.Linear(z,z), nn.LeakyReLU(0.2),
            nn.Linear(z,z), nn.LeakyReLU(0.2),
            nn.Linear(z,z), nn.LeakyReLU(0.2),
            nn.Linear(z,z), nn.LeakyReLU(0.2),
            nn.Linear(z,z), nn.LeakyReLU(0.2),
            F.normalize(z, dim=1)
        )

        self.A11 = nn.Linear(z, ch)
        self.A12 = nn.Linear(z, ch)
        self.A21 = nn.Linear(z, ch)
        self.A22 = nn.Linear(z, ch)
        self.A31 = nn.Linear(z, ch)
        self.A32 = nn.Linear(z, ch)
        self.A41 = nn.Linear(z, ch)
        self.A42 = nn.Linear(z, ch)
        self.A51 = nn.Linear(z, ch)
        self.A52 = nn.Linear(z, ch)
        self.A61 = nn.Linear(z, ch)
        self.A62 = nn.Linear(z, ch)
        self.A71 = nn.Linear(z, ch)
        self.A72 = nn.Linear(z, ch)

        # Noise learnt scalling factors for each latent dim of noise
        self.B11 = nn.Parameter(torch.randn((1)))
        self.B12 = nn.Parameter(torch.randn((1)))
        self.B21 = nn.Parameter(torch.randn((1)))      
        self.B22 = nn.Parameter(torch.randn((1)))      
        self.B31 = nn.Parameter(torch.randn((1)))        
        self.B32 = nn.Parameter(torch.randn((1)))        
        self.B41 = nn.Parameter(torch.randn((1)))        
        self.B42 = nn.Parameter(torch.randn((1)))        
        self.B51 = nn.Parameter(torch.randn((1)))        
        self.B52 = nn.Parameter(torch.randn((1)))        
        self.B61 = nn.Parameter(torch.randn((1))) 
        self.B62 = nn.Parameter(torch.randn((1))) 
        self.B71 = nn.Parameter(torch.randn((1))) 
        self.B72 = nn.Parameter(torch.randn((1))) 

        self.B11b = nn.Parameter(torch.radn((1)))
        self.B12b = nn.Parameter(torch.radn((1)))
        self.B21b = nn.Parameter(torch.radn((1)))
        self.B22b = nn.Parameter(torch.radn((1)))
        self.B31b = nn.Parameter(torch.radn((1)))
        self.B32b = nn.Parameter(torch.radn((1)))
        self.B41b = nn.Parameter(torch.radn((1)))
        self.B42b = nn.Parameter(torch.radn((1)))
        self.B51b = nn.Parameter(torch.radn((1)))
        self.B52b = nn.Parameter(torch.radn((1)))
        self.B61b = nn.Parameter(torch.radn((1)))
        self.B62b = nn.Parameter(torch.radn((1)))
        self.B71b = nn.Parameter(torch.radn((1)))
        self.B72b = nn.Parameter(torch.radn((1)))

        self.up = nn.Upsample(scale_factor=2, align_corners=True, mode='bilinear')
        self.conv41 = nn.Conv2d(ch, ch, 3, 1, 1)
        self.conv81 = nn.Conv2d(ch, ch, 3, 1, 1)
        self.conv82 = nn.Conv2d(ch, ch, 3, 1, 1)
        self.conv161 = nn.Conv2d(ch, ch, 3, 1, 1)
        self.conv162 = nn.Conv2d(ch, ch, 3, 1, 1)
        self.conv321 = nn.Conv2d(ch, ch, 3, 1, 1)
        self.conv322 = nn.Conv2d(ch, ch, 3, 1, 1)
        self.conv641 = nn.Conv2d(ch, ch, 3, 1, 1)
        self.conv642 = nn.Conv2d(ch, ch, 3, 1, 1)
        self.conv1281 = nn.Conv2d(ch, ch, 3, 1, 1)
        self.conv1282 = nn.Conv2d(ch, ch, 3, 1, 1)
        self.conv2561 = nn.Conv2d(ch, ch, 3, 1, 1)
        self.conv2562 = nn.Conv2d(ch, ch, 3, 1, 1)

        self.toRGB = nn.Conv2d(ch, 3, 1, 1)

        self.tanh = nn.Tanh()

    # forward method
    def forward(self, x):
        # x is noise vector
        B = x.shape[0]
        # Style Vectorizer
        w = self.sv(x)

        y11 = self.A11(w)
        y12 = self.A12(w)
        y21 = self.A21(w)
        y22 = self.A22(w)
        y31 = self.A31(w)
        y32 = self.A32(w)
        y41 = self.A41(w)
        y42 = self.A42(w)
        y51 = self.A51(w)
        y52 = self.A52(w)
        y61 = self.A61(w)
        y62 = self.A62(w)
        y71 = self.A71(w)
        y72 = self.A72(w)

        # To-Do: styles Mixing Regularisation
        # To-Do: weights demodulation to remove bubble artifacts

        # Synthesis Network
        # 4 x 4 size
               
        c = self.constants + self.B11*torch.randn((B, 1, 4, 4), device=self.device) + self.B11b
        x = self.AdaIN(c, y11)
        x = self.conv41(x)
        x = x + self.B12*torch.randn((B, 1, 4, 4), device=self.device) + self.B12b
        x = F.leaky_relu(x, 0.2)
        x = self.AdaIN(x, y12 )

        # 8 x 8 size
        x = self.up(x)
        x = self.conv81(x)
        x = x + self.B21*torch.randn((B, 1, 8, 8), device=self.device) + self.B21b
        x = F.leaky_relu(x, 0.2)
        x = self.AdaIN(x, y21)
        x = self.conv82(x)
        x = x + self.B22*torch.randn((B, 1, 8, 8), device=self.device)+ self.B22b
        x = F.leaky_relu(x, 0.2)
        x = self.AdaIN(x, y22)

        # 16 x 16 size
        x = self.up(x)
        x = self.conv161(x)
        x = x + self.B31*torch.randn((B, 1, 16, 16), device=self.device) + self.B31b
        x = F.leaky_relu(x, 0.2)
        x = self.AdaIN(x, y31)
        x = self.conv162(x)
        x = x + self.B32*torch.randn((B, 1, 16, 16), device=self.device) + self.B32b
        x = F.leaky_relu(x, 0.2)
        x = self.AdaIN(x, y32)
        #print(x.shape)

        # 32 x 32 size
        x = self.conv321(x)
        x = self.up(x)
        x = x + self.B41*torch.randn((B, 1, 32, 32), device=self.device) + self.B41b
        x = F.leaky_relu(x, 0.2)
        x = self.AdaIN(x, y41)
        x = self.conv322(x)
        x = x + self.B42*torch.randn((B, 1, 32, 32), device=self.device) + self.B42b
        x = F.leaky_relu(x, 0.2)
        x = self.AdaIN(x, y42)

        # 64 x 64 size
        x = self.conv641(x)
        x = self.up(x)
        x = x + self.B51*torch.randn((B, 1, 64, 64), device=self.device) + self.B51b
        x = F.leaky_relu(x, 0.2)
        x = self.AdaIN(x, y51)
        x = self.conv642(x)
        x = x + self.B52*torch.randn((B, 1, 64, 64), device=self.device) + self.B52b
        x = F.leaky_relu(x, 0.2)
        x = self.AdaIN(x, y52)

        # 128 x 128 size
        x = self.conv1281(x)
        x = self.up(x)
        x = x + self.B61*torch.randn((B, 1, 128, 128), device=self.device) + self.B61b
        x = F.leaky_relu(x, 0.2)
        x = self.AdaIN(x, y61)
        x = self.conv1282(x)
        x = x + self.B62*torch.randn((B, 1, 128, 128), device=self.device) + self.B62b
        x = F.leaky_relu(x, 0.2)
        x = self.AdaIN(x, y62)

        # 256 x 256 size
        x = self.conv2561(x)
        x = self.up(x)
        x = x + self.B71*torch.randn((B, 1, 256, 256), device=self.device)+ self.B71b
        x = F.leaky_relu(x, 0.2)
        x = self.AdaIN(x, y71)
        x = self.conv2562(x)
        x = x + self.B72*torch.randn((B, 1, 256, 256), device=self.device)+ self.B72b
        x = F.leaky_relu(x, 0.2)
        x = self.AdaIN(x, y72)



        x = self.toRGB(x)
        x = self.tanh(x)

        return x
    
class g_stylegan1(nn.Module):
    # initializers
    def __init__(self, z=512, ch=64, img_size=32, gamma=0.01, device='cuda'):
        super().__init__()
        #z+=1 # Adding the class dimension to each random vector
        
        self.AdaIN = AdaIN() # just an operatioin

        self.constants = nn.Parameter(torch.randn((1, ch, 4, 4)))
        self.constants_b = nn.Parameter(torch.randn((1, ch, 1, 1)))
        self.img_size = img_size
        self.device = device
        
        self.sv = nn.Sequential(
            nn.Linear(z,z), nn.LeakyReLU(0.2),
            nn.Linear(z,z), nn.LeakyReLU(0.2),
            nn.Linear(z,z), nn.LeakyReLU(0.2),
            nn.Linear(z,z), nn.LeakyReLU(0.2),
            nn.Linear(z,z), nn.LeakyReLU(0.2),
            nn.Linear(z,z), nn.LeakyReLU(0.2),
            nn.Linear(z,z), nn.LeakyReLU(0.2),
            nn.Linear(z,z), nn.LeakyReLU(0.2),
        )

        self.A11 = nn.Linear(z, ch)
        self.A12 = nn.Linear(z, ch)
        self.A21 = nn.Linear(z, ch)
        self.A22 = nn.Linear(z, ch)
        self.A31 = nn.Linear(z, ch)
        self.A32 = nn.Linear(z, ch)
        self.A41 = nn.Linear(z, ch)
        self.A42 = nn.Linear(z, ch)
        self.A51 = nn.Linear(z, ch)
        self.A52 = nn.Linear(z, ch)
        self.A61 = nn.Linear(z, ch)
        self.A62 = nn.Linear(z, ch)
        self.A71 = nn.Linear(z, ch)
        self.A72 = nn.Linear(z, ch)
        self.A81 = nn.Linear(z, ch)
        self.A82 = nn.Linear(z, ch)
        
        # Noise learnt scalling factors for each latent dim of noise
        self.B11 = nn.Parameter(torch.randn((1, 1, 4, 4)))
        self.B12 = nn.Parameter(torch.randn((1, 1, 4, 4)))
        self.B21 = nn.Parameter(torch.randn((1, 1, 8, 8)))
        self.B22 = nn.Parameter(torch.randn((1, 1, 8, 8)))
        self.B31 = nn.Parameter(torch.randn((1, 1, 16, 16)))
        self.B32 = nn.Parameter(torch.randn((1, 1, 16, 16)))
        self.B41 = nn.Parameter(torch.randn((1, 1, 32, 32)))
        self.B42 = nn.Parameter(torch.randn((1, 1, 32, 32)))
        self.B51 = nn.Parameter(torch.randn((1, 1, 64, 64)))
        self.B52 = nn.Parameter(torch.randn((1, 1, 64, 64)))
        self.B61 = nn.Parameter(torch.randn((1, 1, 128, 128)))
        self.B62 = nn.Parameter(torch.randn((1, 1, 128, 128)))
        self.B71 = nn.Parameter(torch.randn((1, 1, 256, 256)))
        self.B72 = nn.Parameter(torch.randn((1, 1, 256, 256)))
        self.B81 = nn.Parameter(torch.randn((1, 1, 512, 512)))
        self.B82 = nn.Parameter(torch.randn((1, 1, 512, 512)))

        self.B11b = nn.Parameter(torch.randn((1, 1, 1, 1)))
        self.B12b = nn.Parameter(torch.randn((1, 1, 1, 1)))
        self.B21b = nn.Parameter(torch.randn((1, 1, 1, 1)))
        self.B22b = nn.Parameter(torch.randn((1, 1, 1, 1)))
        self.B31b = nn.Parameter(torch.randn((1, 1, 1, 1)))
        self.B32b = nn.Parameter(torch.randn((1, 1, 1, 1)))
        self.B41b = nn.Parameter(torch.randn((1, 1, 1, 1)))
        self.B42b = nn.Parameter(torch.randn((1, 1, 1, 1)))
        self.B51b = nn.Parameter(torch.randn((1, 1, 1, 1)))
        self.B52b = nn.Parameter(torch.randn((1, 1, 1, 1)))
        self.B61b = nn.Parameter(torch.randn((1, 1, 1, 1)))
        self.B62b = nn.Parameter(torch.randn((1, 1, 1, 1)))
        self.B71b = nn.Parameter(torch.randn((1, 1, 1, 1)))
        self.B72b = nn.Parameter(torch.randn((1, 1, 1, 1)))
        self.B81b = nn.Parameter(torch.randn((1, 1, 1, 1)))
        self.B82b = nn.Parameter(torch.randn((1, 1, 1, 1)))
        
        self.up = nn.Upsample(scale_factor=2, align_corners=True, mode='bilinear')
        self.conv41 = nn.Conv2d(ch, ch, 3, 1, 1)
        self.conv81 = nn.Conv2d(ch, ch, 3, 1, 1)
        self.conv82 = nn.Conv2d(ch, ch, 3, 1, 1)
        self.conv161 = nn.Conv2d(ch, ch, 3, 1, 1)
        self.conv162 = nn.Conv2d(ch, ch, 3, 1, 1)
        self.conv321 = nn.Conv2d(ch, ch, 3, 1, 1)
        self.conv322 = nn.Conv2d(ch, ch, 3, 1, 1)
        self.conv641 = nn.Conv2d(ch, ch, 3, 1, 1)
        self.conv642 = nn.Conv2d(ch, ch, 3, 1, 1)
        self.conv1281 = nn.Conv2d(ch, ch, 3, 1, 1)
        self.conv1282 = nn.Conv2d(ch, ch, 3, 1, 1)
        self.conv2561 = nn.Conv2d(ch, ch, 3, 1, 1)
        self.conv2562 = nn.Conv2d(ch, ch, 3, 1, 1)
        self.conv5121 = nn.Conv2d(ch, ch, 3, 1, 1)
        self.conv5122 = nn.Conv2d(ch, ch, 3, 1, 1)
        
        self.toRGB = nn.Conv2d(ch, 3, 1, 1)

        self.tanh = nn.Tanh()

    # forward method
    def forward(self, x):
        # x is noise vector
        B = x.shape[0]
        # Style Vectorizer
        w = self.sv(x)

        y11 = self.A11(w)
        y12 = self.A12(w)
        y21 = self.A21(w)
        y22 = self.A22(w)
        y31 = self.A31(w)
        y32 = self.A32(w)
        y41 = self.A41(w)
        y42 = self.A42(w)
        y51 = self.A51(w)
        y52 = self.A52(w)
        y61 = self.A61(w)
        y62 = self.A62(w)
        y71 = self.A71(w)
        y72 = self.A72(w)
        y81 = self.A81(w)
        y82 = self.A82(w)
        
        # To-Do: styles Mixing Regularisation
        # To-Do: weights demodulation to remove bubble artifacts

        # Synthesis Network
        # 4 x 4 size
               
        c = self.constants + self.B11*torch.randn((B, 1, 4, 4), device=self.device) + self.B11b
        x = self.AdaIN(c, y11)
        x = self.conv41(x)
        x = x + self.B12*torch.randn((B, 1, 4, 4), device=self.device) + self.B12b
        x = F.leaky_relu(x, 0.2)
        x = self.AdaIN(x, y12 )

        # 8 x 8 size
        x = self.up(x)
        x = self.conv81(x)
        x = x + self.B21*torch.randn((B, 1, 8, 8), device=self.device) + self.B21b
        x = F.leaky_relu(x, 0.2)
        x = self.AdaIN(x, y21)
        x = self.conv82(x)
        x = x + self.B22*torch.randn((B, 1, 8, 8), device=self.device) + self.B22b
        x = F.leaky_relu(x, 0.2)
        x = self.AdaIN(x, y22)

        # 16 x 16 size
        x = self.up(x)
        x = self.conv161(x)
        x = x + self.B31*torch.randn((B, 1, 16, 16), device=self.device) + self.B31b
        x = F.leaky_relu(x, 0.2)
        x = self.AdaIN(x, y31)
        x = self.conv162(x)
        x = x + self.B32*torch.randn((B, 1, 16, 16), device=self.device) + self.B32b
        x = F.leaky_relu(x, 0.2)
        x = self.AdaIN(x, y32)
        #print(x.shape)

        # 32 x 32 size
        x = self.conv321(x)
        x = self.up(x)
        x = x + self.B41*torch.randn((B, 1, 32, 32), device=self.device) + self.B41b
        x = F.leaky_relu(x, 0.2)
        x = self.AdaIN(x, y41)
        x = self.conv322(x)
        x = x + self.B42*torch.randn((B, 1, 32, 32), device=self.device) + self.B42b
        x = F.leaky_relu(x, 0.2)
        x = self.AdaIN(x, y42)
        
        if self.img_size==32:
            return self.tanh(self.toRGB(x))
        
        # 64 x 64 size
        x = self.conv641(x)
        x = self.up(x)
        x = x + self.B51*torch.randn((B, 1, 64, 64), device=self.device) + self.B51b
        x = F.leaky_relu(x, 0.2)
        x = self.AdaIN(x, y51)
        x = self.conv642(x)
        x = x + self.B52*torch.randn((B, 1, 64, 64), device=self.device) + self.B52b
        x = F.leaky_relu(x, 0.2)
        x = self.AdaIN(x, y52)
        
        if self.img_size==64:
            return self.tanh(self.toRGB(x))
        
        # 128 x 128 size
        x = self.conv1281(x)
        x = self.up(x)
        x = x + self.B61*torch.randn((B, 1, 128, 128), device=self.device) + self.B61b
        x = F.leaky_relu(x, 0.2)
        x = self.AdaIN(x, y61)
        x = self.conv1282(x)
        x = x + self.B62*torch.randn((B, 1, 128, 128), device=self.device) + self.B62b
        x = F.leaky_relu(x, 0.2)
        x = self.AdaIN(x, y62)
        
        if self.img_size==128:
            return self.tanh(self.toRGB(x))
        
        # 256 x 256 size
        x = self.conv2561(x)
        x = self.up(x)
        x = x + self.B71*torch.randn((B, 1, 256, 256), device=self.device) + self.B71b
        x = F.leaky_relu(x, 0.2)
        x = self.AdaIN(x, y71)
        x = self.conv2562(x)
        x = x + self.B72*torch.randn((B, 1, 256, 256), device=self.device) + self.B72b
        x = F.leaky_relu(x, 0.2)
        x = self.AdaIN(x, y72)
        
        if self.img_size==256:
            return self.tanh(self.toRGB(x))
        
        # 512 x 512 size
        x = self.conv5121(x)
        x = self.up(x)
        x = x + self.B81*torch.randn((B, 1, 512, 512), device=self.device) + self.B81b
        x = F.leaky_relu(x, 0.2)
        x = self.AdaIN(x, y81)
        x = self.conv5122(x)
        x = x + self.B82*torch.randn((B, 1, 512, 512), device=self.device) + self.B82b
        x = F.leaky_relu(x, 0.2)
        x = self.AdaIN(x, y82)
        
       # x = self.tanh(x)

        return self.tanh(self.toRGB(x))