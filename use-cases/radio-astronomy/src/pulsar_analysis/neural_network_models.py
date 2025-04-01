import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np 
from collections import OrderedDict


class UNet(nn.Module):
    """_summary_

    :param nn: _description_
    :type nn: _type_
    """

    def __init__(self, in_channels=1, out_channels=1, init_features=32):
        super(UNet, self).__init__()
        features = init_features
        self.encoder1 = UNet._block(
            in_channels, features, name="enc1"
        )
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.encoder2 = UNet._block(
            features, features * 2, name="enc2"
        )
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.encoder3 = UNet._block(
            features * 2, features * 4, name="enc3"
        )
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.encoder4 = UNet._block(
            features * 4, features * 8, name="enc4"
        )
        self.pool4 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.bottleneck = UNet._block(
            features * 8, features * 16, name="bottleneck"
        )

        self.upconv4 = nn.ConvTranspose2d(
            features * 16, features * 8, kernel_size=2, stride=2
        )
        self.decoder4 = UNet._block(
            (features * 8) * 2, features * 8, name="dec4"
        )
        self.upconv3 = nn.ConvTranspose2d(
            features * 8, features * 4, kernel_size=2, stride=2
        )
        self.decoder3 = UNet._block(
            (features * 4) * 2, features * 4, name="dec3"
        )
        self.upconv2 = nn.ConvTranspose2d(
            features * 4, features * 2, kernel_size=2, stride=2
        )
        self.decoder2 = UNet._block(
            (features * 2) * 2, features * 2, name="dec2"
        )
        self.upconv1 = nn.ConvTranspose2d(
            features * 2, features, kernel_size=2, stride=2
        )
        self.decoder1 = UNet._block(
            features * 2, features, name="dec1"
        )

        self.conv = nn.Conv2d(
            in_channels=features, out_channels=out_channels, kernel_size=1
        )

        self.sigmoid_layer = nn.Sigmoid()

    def forward(self, x):
        enc1 = self.encoder1(x)
        enc2 = self.encoder2(self.pool1(enc1))
        enc3 = self.encoder3(self.pool2(enc2))
        enc4 = self.encoder4(self.pool3(enc3))

        bottleneck = self.bottleneck(self.pool4(enc4))

        dec4 = self.upconv4(bottleneck)
        dec4 = torch.cat((dec4, enc4), dim=1)
        dec4 = self.decoder4(dec4)
        dec3 = self.upconv3(dec4)
        dec3 = torch.cat((dec3, enc3), dim=1)
        dec3 = self.decoder3(dec3)
        dec2 = self.upconv2(dec3)
        dec2 = torch.cat((dec2, enc2), dim=1)
        dec2 = self.decoder2(dec2)
        dec1 = self.upconv1(dec2)
        dec1 = torch.cat((dec1, enc1), dim=1)
        dec1 = self.decoder1(dec1)
        #dec1 = self.sigmoid_layer(dec1)
        return torch.sigmoid(self.conv(dec1))
        

    @staticmethod
    def _block(in_channels, features, name):
        return nn.Sequential(
            OrderedDict(
                [
                    (
                        name + "conv1",
                        nn.Conv2d(
                            in_channels=in_channels,
                            out_channels=features,
                            kernel_size=3,
                            padding=1,
                            bias=False,
                        ),
                    ),
                    (name + "norm1", nn.BatchNorm2d(num_features=features)),
                    (name + "relu1", nn.ReLU(inplace=True)),
                    (
                        name + "conv2",
                        nn.Conv2d(
                            in_channels=features,
                            out_channels=features,
                            kernel_size=3,
                            padding=1,
                            bias=False,
                        ),
                    ),
                    (name + "norm2", nn.BatchNorm2d(num_features=features)),
                    (name + "relu2", nn.ReLU(inplace=True)),
                ]
            )
        )

class FilterCNN(nn.Module):
    def __init__(self, in_channels=1, out_channels=1, init_features=32):
        super(FilterCNN, self).__init__()
        features = init_features
        self.encoder1 = FilterCNN._block(
            in_channels, features,kernel_size=7, name="enc1"
        )
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.encoder2 = FilterCNN._block(
            features, features * 2,kernel_size=7, name="enc2"
        )
        
        self.decoder2 = FilterCNN._block(
            (features * 2) * 1, features * 2,kernel_size=7, name="dec2"
        )
        self.upconv1 = nn.ConvTranspose2d(
            features * 2, features, kernel_size=2, stride=2
        )
        self.decoder1 = FilterCNN._block(
            features * 1, features,kernel_size=7, name="dec1"
        )

        self.conv = nn.Conv2d(
            in_channels=features, out_channels=out_channels, kernel_size=1
        )

        

    def forward(self, x):
        enc1 = self.encoder1(x)
        enc1 = self.pool1(enc1)

        enc2 = self.encoder2(enc1)        
        dec2 = self.decoder2(enc2)
        
        dec1 = self.upconv1(dec2)        
        dec1 = self.decoder1(dec1)        
        return torch.sigmoid(self.conv(dec1))

    @staticmethod
    def _block(in_channels, features, name,kernel_size:int):
        return nn.Sequential(
            OrderedDict(
                [
                    (
                        name + "conv1",
                        nn.Conv2d(
                            in_channels=in_channels,
                            out_channels=features,
                            kernel_size=kernel_size,
                            padding=int(kernel_size/2),
                            bias=False,
                        ),
                    ),
                    (name + "norm1", nn.BatchNorm2d(num_features=features)),
                    (name + "relu1", nn.ReLU(inplace=True)),
                    (
                        name + "conv2",
                        nn.Conv2d(
                            in_channels=features,
                            out_channels=features,
                            kernel_size=5,
                            padding=2,
                            bias=False,
                        ),
                    ),
                    (name + "norm2", nn.BatchNorm2d(num_features=features)),
                    (name + "relu2", nn.ReLU(inplace=True)),
                ]
            )
        )
    
class UNetFilter(nn.Module):
    """_summary_

    :param nn: _description_
    :type nn: _type_
    """

    def __init__(self, in_channels=1, out_channels=1, init_features=32):
        super(UNetFilter, self).__init__()
        features = init_features
        self.encoder1 = UNetFilter._block(
            in_channels, features,kernel_size=7, name="enc1"
        )
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.encoder2 = UNetFilter._block(
            features, features * 2,kernel_size=7, name="enc2"
        )
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.encoder3 = UNetFilter._block(
            features * 2, features * 4,kernel_size=7, name="enc3"
        )
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.encoder4 = UNetFilter._block(
            features * 4, features * 8,kernel_size=7, name="enc4"
        )
        self.pool4 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.bottleneck = UNetFilter._block(
            features * 8, features * 16,kernel_size=7, name="bottleneck"
        )

        self.upconv4 = nn.ConvTranspose2d(
            features * 16, features * 8, kernel_size=2, stride=2
        )
        self.decoder4 = UNetFilter._block(
            (features * 8) * 2, features * 8,kernel_size=7, name="dec4"
        )
        self.upconv3 = nn.ConvTranspose2d(
            features * 8, features * 4, kernel_size=2, stride=2
        )
        self.decoder3 = UNetFilter._block(
            (features * 4) * 2, features * 4,kernel_size=7, name="dec3"
        )
        self.upconv2 = nn.ConvTranspose2d(
            features * 4, features * 2, kernel_size=2, stride=2
        )
        self.decoder2 = UNetFilter._block(
            (features * 2) * 2, features * 2,kernel_size=7, name="dec2"
        )
        self.upconv1 = nn.ConvTranspose2d(
            features * 2, features, kernel_size=2, stride=2
        )
        self.decoder1 = UNetFilter._block(
            features * 2, features,kernel_size=7, name="dec1"
        )

        self.conv = nn.Conv2d(
            in_channels=features, out_channels=out_channels, kernel_size=1
        )

        

    def forward(self, x):
        enc1 = self.encoder1(x)
        enc2 = self.encoder2(self.pool1(enc1))
        enc3 = self.encoder3(self.pool2(enc2))
        enc4 = self.encoder4(self.pool3(enc3))

        bottleneck = self.bottleneck(self.pool4(enc4))

        dec4 = self.upconv4(bottleneck)
        dec4 = torch.cat((dec4, enc4), dim=1)
        dec4 = self.decoder4(dec4)
        dec3 = self.upconv3(dec4)
        dec3 = torch.cat((dec3, enc3), dim=1)
        dec3 = self.decoder3(dec3)
        dec2 = self.upconv2(dec3)
        dec2 = torch.cat((dec2, enc2), dim=1)
        dec2 = self.decoder2(dec2)
        dec1 = self.upconv1(dec2)
        dec1 = torch.cat((dec1, enc1), dim=1)
        dec1 = self.decoder1(dec1)        
        return torch.sigmoid(self.conv(dec1))
        

    @staticmethod
    def _block(in_channels, features, name,kernel_size:int):
        return nn.Sequential(
            OrderedDict(
                [
                    (
                        name + "conv1",
                        nn.Conv2d(
                            in_channels=in_channels,
                            out_channels=features,
                            kernel_size=kernel_size,
                            padding=int(kernel_size/2),
                            bias=False,
                        ),
                    ),
                    (name + "norm1", nn.BatchNorm2d(num_features=features)),
                    (name + "relu1", nn.ReLU(inplace=True)),
                    (
                        name + "conv2",
                        nn.Conv2d(
                            in_channels=features,
                            out_channels=features,
                            kernel_size=5,
                            padding=2,
                            bias=False,
                        ),
                    ),
                    (name + "norm2", nn.BatchNorm2d(num_features=features)),
                    (name + "relu2", nn.ReLU(inplace=True)),
                ]
            )
        )
    

class CustomLossUNet(nn.Module):
    def __init__(self):
        super(CustomLossUNet, self).__init__()

    def forward(self, output, target):        
        loss = (output - target) ** 2
        mask = target >= 0
        high_cost = (loss * mask.float()).mean()
        return high_cost
    
class CustomLossSemanticSeg(nn.Module):
    def __init__(self, smooth=1e-6):
        super(CustomLossSemanticSeg, self).__init__()
        self.smooth = smooth

    def forward(self, preds, targets):
        #preds = (preds > 0.5).float()
        preds = preds.contiguous()
        targets = targets.contiguous()
        
        intersection = (preds * targets).sum(dim=2).sum(dim=2)
        dice_coef = (2. * intersection + self.smooth) / (preds.sum(dim=2).sum(dim=2) + targets.sum(dim=2).sum(dim=2) + self.smooth)
        
        return 1 - dice_coef.mean()
    
class OneDconvEncoder(nn.Module):

    def __init__(self, in_channels=1, out_classes=2, init_features=8, sig_dim=128): #init_features=32
        super(OneDconvEncoder, self).__init__()
        features = init_features

        self.encoder1 = OneDconvEncoder._block(
            in_channels, features, name="enc1"
        )
        self.pool1 = nn.MaxPool1d(kernel_size=2, stride=2)

        self.encoder2 = OneDconvEncoder._block(
            features, features * 2, name="enc2"
        )
        self.pool2 = nn.MaxPool1d(kernel_size=2, stride=2)

        self.encoder3 = OneDconvEncoder._block(
            features * 2, features * 4, name="enc3"
        )
        self.pool3 = nn.MaxPool1d(kernel_size=2, stride=2)

        self.encoder4 = OneDconvEncoder._block(
            features * 4, features * 8, name="enc4"
        )
        self.pool4 = nn.MaxPool1d(kernel_size=2, stride=2)

        # output dimension features*8 x image_dim/16 
        self.linear_encoder = OneDconvEncoder._to_linear_block(sig_dim=int(np.ceil(sig_dim/2)),features=features*2,out_classes=out_classes) 

        

    def forward(self, x):
        enc1 = self.encoder1(x)
        #print('debug: shape',enc1.shape)
        enc2 = self.encoder2(self.pool1(enc1))
        #print('debug: shape',enc2.shape)
        #enc3 = self.encoder3(self.pool2(enc2))
        #print('debug: shape',enc3.shape)
        #enc4 = self.encoder4(self.pool3(enc3))
        #print('debug: shape',enc4.shape)
        lenc1 = self.linear_encoder(enc2)
        return lenc1

    
    @staticmethod
    def _block(in_channels, features, name):
        return nn.Sequential(
            OrderedDict(
                [
                    (
                        name + "conv1",
                        nn.Conv1d(
                            in_channels=in_channels,
                            out_channels=features,
                            kernel_size=7,
                            padding=3,
                            bias=False,
                        ),
                    ),
                    (name + "norm1", nn.BatchNorm1d(num_features=features)),
                    (name + "relu1", nn.ReLU(inplace=True)),
                    (name + "dropout1", nn.Dropout(0.5)),
                    (
                        name + "conv2",
                        nn.Conv1d(
                            in_channels=features,
                            out_channels=features,
                            kernel_size=5,
                            padding=2,
                            bias=False,
                        ),
                    ),
                    (name + "norm2", nn.BatchNorm1d(num_features=features)),
                    (name + "relu2", nn.ReLU(inplace=True)),
                    (name + "dropout2", nn.Dropout(0.5)),
                ]
            )
        )
    pass

    @staticmethod
    def _to_linear_block(sig_dim,features,out_classes):
        network = nn.Sequential(
            nn.Flatten(),            
            nn.Linear(features*sig_dim,128),
            nn.ReLU(),
            nn.Linear(128,64),
            nn.ReLU(),
            nn.Linear(64,out_classes),
            nn.Softmax(dim=1))
        return network


class Simple1DCnnClassifier(nn.Module):
    def __init__(self,init_features:int=8,input_signal_len:int = 128):
        super(Simple1DCnnClassifier, self).__init__()
        self.cnn_block_0 = Simple1DCnnClassifier.cnn_unit_block(in_channels=1,features=init_features,kernel_len=5) #: output [init_featuresx128]
        self.pool0 = nn.MaxPool1d(kernel_size=2, stride=2) #: output [init_featuresx64]
        self.cnn_block_1 = Simple1DCnnClassifier.cnn_unit_block(in_channels=init_features,features=int(init_features*4),kernel_len=5)  #: output [init_features*2x64]
        self.pool1 = nn.MaxPool1d(kernel_size=2, stride=2) #: output [init_features*2x32]
        self.cnn_block_2 = Simple1DCnnClassifier.cnn_unit_block(in_channels=init_features*4,features=init_features*8,kernel_len=5) #: output [init_features*2x32]
        self.pool2 = nn.MaxPool1d(kernel_size=2, stride=2) #: output [init_features*2x16]
        self.flatten_block = nn.Flatten()
        self.fully_connected_block_0 = Simple1DCnnClassifier.fully_connected_unit_block(in_channels=int(input_signal_len/8*init_features*8),out_features=512)
        self.fully_connected_block_1 = Simple1DCnnClassifier.fully_connected_unit_block(in_channels=512,out_features=256)
        self.fully_connected_block_2 = Simple1DCnnClassifier.fully_connected_unit_block(in_channels=256,out_features=128)  
        self.fully_connected_block_3 = Simple1DCnnClassifier.fully_connected_unit_block(in_channels=128,out_features=64) 
        self.fully_connected_block_4 = Simple1DCnnClassifier.fully_connected_unit_block(in_channels=64,out_features=1)        
        self.sigmoid_layer = nn.Sigmoid()
        #self.softmax_layer = nn.Softmax(dim=1)

    def forward(self, x):
        x = self.cnn_block_0(x)
        x = self.pool0(x)
        #print('cnn0_out',cnn0_out.shape)
        x = self.cnn_block_1(x)
        x = self.pool1(x)
        #print('cnn1_out',cnn1_out.shape)
        x = self.cnn_block_2(x)
        x = self.pool2(x)
        x = self.flatten_block(x)
        #print('flattened_output ',flattened_output .shape)
        x= self.fully_connected_block_0(x)
        #print('fully_nn_out_0 ',fully_nn_out_0 .shape)
        x = self.fully_connected_block_1(x)
        x = self.fully_connected_block_2(x)
        x = torch.relu(x)
        x = self.fully_connected_block_3(x)
        x = torch.relu(x)
        x = self.fully_connected_block_4(x)
        #normalized_output = self.softmax_layer(fully_nn_out_2)
        x = self.sigmoid_layer(x)
        #normalized_output = self.softmax_layer(normalized_output)
        #print('normalized_output',normalized_output)
        return x

    @staticmethod
    def cnn_unit_block(features:int,kernel_len:int=11,in_channels:int = 1):
        block = nn.Sequential(
            nn.Conv1d(in_channels,features,kernel_len,padding=int(kernel_len/2)),
            #nn.Conv1d(in_channels=in_channels,out_channels=features,kernel_size=kernel_len,padding=kernel_len/2,bias=False),
            #nn.BatchNorm1d(num_features=features),
            nn.ReLU(inplace=True),
            #nn.Conv1d(features,features,kernel_len,padding=int(kernel_len/2)),
            #nn.BatchNorm1d(num_features=features),
            #nn.ReLU(inplace=True),
            nn.Dropout(p=0.1)
        )
        return block

    @staticmethod
    def fully_connected_unit_block(in_channels:int,out_features:int):
        block = nn.Sequential(
            nn.Linear(in_features=in_channels,out_features=out_features),
            #nn.BatchNorm1d(num_features=out_features),
            #nn.LayerNorm(out_features),
            #nn.ReLU(inplace=True),            
            nn.Dropout(p=0.2),            
        )
        return block
    
class CNN1D(nn.Module):
    def __init__(self, input_channels=1, num_classes=1,sig_len=128):  # num_classes set to 1 for binary classification
        super(CNN1D, self).__init__()
        
        self.conv1 = nn.Conv1d(in_channels=input_channels, out_channels=16, kernel_size=3, stride=1, padding=1) #sig_lenx16
        self.conv2 = nn.Conv1d(in_channels=16, out_channels=32, kernel_size=3, stride=1, padding=1) #sig_lenx32
        self.conv3 = nn.Conv1d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1)#sig_lenx64
        
        self.flatten_method = nn.Flatten()

        self.fc1 = nn.Linear(int(64 * sig_len/8), 128)  # Assuming input sequence length is 8
        self.fc2 = nn.Linear(128, num_classes)
        
        self.pool = nn.MaxPool1d(kernel_size=2, stride=2, padding=0)
        self.dropout = nn.Dropout(0.5)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x))) #sig_len/2x16
        x = self.pool(F.relu(self.conv2(x))) #sig_len/4x32
        x = self.pool(F.relu(self.conv3(x))) #sig_len/8x64
        x = self.flatten_method(x)        
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = torch.sigmoid(self.fc2(x))  # Sigmoid activation for binary output
        return x
    

class WeightedBCELoss(nn.Module):
    def __init__(self, pos_weight=1.0, neg_weight=1.0,eps=1e-7):
        super(WeightedBCELoss, self).__init__()
        self.pos_weight = pos_weight
        self.neg_weight = neg_weight
        self.eps = eps 

    def forward(self, inputs, targets):        
        inputs = torch.clamp(inputs, self.eps, 1.0 - self.eps)        
        loss = - (self.pos_weight * targets * torch.log(inputs) + 
                  self.neg_weight * (1 - targets) * torch.log(1 - inputs)).mean()
        
        return loss
class CustomLossClassifier(nn.Module):
    def __init__(self):
        super(CustomLossClassifier, self).__init__()

    def forward(self, output, target):
        #print('output',output,'target',target)        
        loss = ((output - target)**2).mean()
        #mask = target >= 0
        #high_cost = (loss * mask.float()).mean()
        return loss
    
