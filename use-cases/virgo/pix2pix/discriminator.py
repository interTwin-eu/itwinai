import torch
import torch.nn as nn
from generator import downsample


class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.down1 = downsample(6, 64, apply_batchnorm=False)
        self.down2 = downsample(64, 128)
        self.down3 = downsample(128, 256)
        
        self.conv1 = nn.Conv2d(256, 512, 4, 1, 1, bias=False)
        self.batchnorm = nn.BatchNorm2d(512)
        self.leakyrelu = nn.LeakyReLU(inplace=True)
        
        self.last = nn.Sequential()
        self.last.add_module(name="Conv2d", module=nn.Conv2d(512, 1, 4, 1, 1))
        # self.last.add_module(name="sigmoid", module=nn.Sigmoid())
    
    def forward(self, inp, tar):
        x = torch.cat([inp, tar], dim=1)
        x = self.down1(x)
        x = self.down2(x)
        x = self.down3(x)
        
        x = self.conv1(x)
        x = self.batchnorm(x)
        x = self.leakyrelu(x)
        
        x = self.last(x)
        
        return x