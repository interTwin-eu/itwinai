import torch
import torch.nn as nn

OUTPUT_CHANNELS = 3


def downsample(in_c, out_c, apply_batchnorm=True):
    result = nn.Sequential()
    result.add_module(name="Conv2d", module=nn.Conv2d(in_c, out_c, 4, 2, 1, bias=False))
    if apply_batchnorm:
        result.add_module(name="BatchNorm2d", module=nn.BatchNorm2d(out_c))
    result.add_module(name="LeakyReLU", module=nn.LeakyReLU(inplace=True))

    return result


def upsample(in_c, out_c, apply_dropout=False):
    result = nn.Sequential()
    result.add_module(name="ConvTranspose2d", module=nn.ConvTranspose2d(in_c, out_c, 4, 2, 1, bias=False))
    result.add_module(name="BatchNorm2d", module=nn.BatchNorm2d(out_c))
    if apply_dropout:
        result.add_module(name="Dropout", module=nn.Dropout(0.5, inplace=True))
    result.add_module(name="ReLU", module=nn.ReLU(inplace=True))

    return result


class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()
        self.down1 = downsample(3, 64, apply_batchnorm=False)
        self.down2 = downsample(64, 128)
        self.down3 = downsample(128, 256)
        self.down4 = downsample(256, 512)
        self.down5_7 = downsample(512, 512)
        self.down8 = downsample(512, 512, apply_batchnorm=False)

        self.up1 = upsample(512, 512, apply_dropout=True)
        self.up2_3 = upsample(1024, 512, apply_dropout=True)
        self.up4 = upsample(1024, 512)
        self.up5 = upsample(1024, 256)
        self.up6 = upsample(512, 128)
        self.up7 = upsample(256, 64)

        self.last = nn.Sequential()
        self.last.add_module(name="ConvTranspose2d", module=nn.ConvTranspose2d(128, OUTPUT_CHANNELS, 4, 2, 1))
        self.last.add_module(name="tanh", module=nn.Tanh())

    def forward(self, image):
        # Encoder
        x1 = self.down1(image)
        x2 = self.down2(x1)
        x3 = self.down3(x2)
        x4 = self.down4(x3)
        x5 = self.down5_7(x4)
        x6 = self.down5_7(x5)
        x7 = self.down5_7(x6)
        x8 = self.down8(x7)

        # Decoder
        x = self.up1(x8)
        x = torch.cat([x7, x], dim=1)
        x = self.up2_3(x)
        x = torch.cat([x6, x], dim=1)
        x = self.up2_3(x)
        x = torch.cat([x5, x], dim=1)
        x = self.up4(x)
        x = torch.cat([x4, x], dim=1)
        x = self.up5(x)
        x = torch.cat([x3, x], dim=1)
        x = self.up6(x)
        x = torch.cat([x2, x], dim=1)
        x = self.up7(x)
        x = torch.cat([x1, x], dim=1)

        x = self.last(x)

        return x


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
