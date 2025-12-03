"""
In this section we define different NN architectures models, and initialise
one of them as the generator to use in training and inference.

This section is split in three parts:
- 1) **Weight Initialization**, where we define the function to initialise the
    weights of the NN models according to certain parameters and distributions
    given as input.
- 2) **NN Models**, where we define different NN models exploiting different
    architectures.
- 3) **Generator**, where we initialise one of the above models as the
    generator to use in training and inference
"""

import torch
import torch.nn as nn

# SHALLOW DECODER


class Decoder(nn.Module):
    """
    Decoder network.

    Args:
        - in_channels (int): Number of input channels.
        - kernel_size (int): Size of the convolutional kernel (default is 7).
        - a (float): Scaling factor (default is 80.0).
        - norm (bool): Whether to apply normalization (default is True).
    """

    def __init__(self, in_channels, kernel_size=7, norm=True):
        super(Decoder, self).__init__()

        self.conv1 = nn.Conv2d(
            in_channels, 32, kernel_size=kernel_size, stride=1,
            padding=kernel_size // 2)
        self.relu1 = nn.LeakyReLU(0.2, inplace=True)

        self.conv2 = nn.Conv2d(
            32, 64, kernel_size=kernel_size, stride=1,
            padding=kernel_size // 2)
        self.relu2 = nn.LeakyReLU(0.2, inplace=True)

        self.conv3 = nn.Conv2d(
            64, 64, kernel_size=kernel_size, stride=1,
            padding=kernel_size // 2)
        self.relu3 = nn.LeakyReLU(0.2, inplace=True)

        self.conv4 = nn.Conv2d(
            64, 1, kernel_size=kernel_size, stride=1,
            padding=kernel_size // 2)

        if norm:
            self.activation = torch.nn.Sigmoid()
        else:
            self.activation = torch.nn.ReLU()

    def _forward_features(self, x):
        """
        Perform forward pass through the network layers.

        Args:
            - x (torch.Tensor): Input tensor.

        Returns:
            - torch.Tensor: Output tensor.
        """
        x = self.relu1(self.conv1(x))
        x = self.relu2(self.conv2(x))
        x = self.relu3(self.conv3(x))
        x = self.conv4(x)
        x = self.activation(x)
        return x

    def forward(self, x):
        """
        Forward pass through the network.

        Args:
            - x (torch.Tensor): Input tensor.

        Returns:
            - torch.Tensor: Output tensor.
        """
        return self._forward_features(x)


# DEEP DECODER


class Decoder_2d_deep(nn.Module):
    """
    Deep 2D decoder network.

    Args:
        - in_channels (int): Number of input channels.
        - kernel_size (int): Size of the convolutional kernel (default is 5).
    """

    def __init__(self, in_channels, kernel_size=5, norm=True):
        super(Decoder_2d_deep, self).__init__()

        self.conv1 = nn.Conv2d(
            in_channels, 64, kernel_size=kernel_size, stride=1,
            padding=kernel_size // 2)
        self.relu1 = nn.LeakyReLU(0.3, inplace=True)

        self.conv2 = nn.Conv2d(
            64, 128, kernel_size=kernel_size, stride=1,
            padding=kernel_size // 2)
        self.relu2 = nn.LeakyReLU(0.3, inplace=True)

        self.conv3 = nn.Conv2d(
            128, 256, kernel_size=kernel_size, stride=1,
            padding=kernel_size // 2)
        self.relu3 = nn.LeakyReLU(0.3, inplace=True)

        self.conv4 = nn.Conv2d(
            256, 1, kernel_size=kernel_size, stride=1,
            padding=kernel_size // 2)
        if norm:
            self.activation = torch.nn.Sigmoid()
        else:
            self.activation = torch.nn.ReLU()

    def _forward_features(self, x):
        """
        Perform forward pass through the network layers.

        Args:
            - x (torch.Tensor): Input tensor.

        Returns:
            - torch.Tensor: Output tensor.
        """
        x = self.relu1(self.conv1(x))
        x = self.relu2(self.conv2(x))
        x = self.relu3(self.conv3(x))
        x = self.conv4(x)
        x = self.activation(x)
        return x

    def forward(self, x):
        """
        Forward pass through the network.

        Args:
            - x (torch.Tensor): Input tensor.

        Returns:
            - torch.Tensor: Output tensor.
        """
        return self._forward_features(x)


# RESNET


class ResidualBlock(nn.Module):
    """
    Residual block module.

    Args:
        - in_features (int): Number of input features/channels.
    """

    def __init__(self, in_features):
        super(ResidualBlock, self).__init__()

        # Define the block sequence
        self.block = nn.Sequential(
            # Pads the input tensor using the reflection of the input boundary
            nn.ReflectionPad2d(1),
            nn.Conv2d(in_features, in_features, 3),  # 2D convolutional layer
            nn.InstanceNorm2d(in_features),  # Instance normalization
            nn.ReLU(inplace=True),  # ReLU activation function
            nn.ReflectionPad2d(1),
            nn.Conv2d(in_features, in_features, 3),
            nn.InstanceNorm2d(in_features)
        )

    def forward(self, x):
        """
        Forward pass through the residual block.

        Args:
            - x (torch.Tensor): Input tensor.

        Returns:
            - torch.Tensor: Output tensor.
        """
        return x + self.block(x)


class GeneratorResNet(nn.Module):
    """
    Generator network using ResNet architecture.

    Args:
        - input_shape (int): Number of input features/channels.
        - num_residual_block (int): Number of residual blocks.
        - output_shape (int): Number of output features/channels.
    """

    def __init__(
            self, input_shape, num_residual_block, output_shape, norm=False
    ):
        super(GeneratorResNet, self).__init__()

        channels = input_shape
        target_channels = output_shape
        # Initial Convolution Block
        out_features = 64
        if norm:
            self.final_activation = torch.nn.Sigmoid()
        else:
            self.final_activation = torch.nn.ReLU()

        model = [
            nn.ReflectionPad2d(channels),
            nn.Conv2d(channels, out_features, 7),
            nn.InstanceNorm2d(out_features),
            nn.ReLU(inplace=True)
        ]
        in_features = out_features

        # Downsampling
        for _ in range(2):
            out_features *= 2
            model += [
                nn.Conv2d(in_features, out_features, 3, stride=2, padding=1),
                nn.InstanceNorm2d(out_features),
                nn.ReLU(inplace=True)
            ]
            in_features = out_features

        # Residual blocks
        for _ in range(num_residual_block):
            model += [ResidualBlock(out_features)]

        # Upsampling
        for _ in range(2):
            out_features //= 2
            model += [
                nn.Upsample(scale_factor=2),  # Upsampling layer
                nn.Conv2d(in_features, out_features, 3, stride=1, padding=1),
                nn.ReLU(inplace=True)
            ]
            in_features = out_features

        # Output Layer
        model += [nn.ReflectionPad2d(target_channels),
                  nn.Conv2d(out_features, target_channels, 3),
                  self.final_activation  # Sigmoid activation function
                  ]

        # Unpacking
        self.model = nn.Sequential(*model)

    def forward(self, x):
        """
        Forward pass through the generator network.

        Args:
            - x (torch.Tensor): Input tensor.

        Returns:
            - torch.Tensor: Output tensor.
        """
        return self.model(x)


# U-NET


class Conv2dBlock(nn.Module):
    """
    Convolutional block module.

    Args:
        - in_channels (int): Number of input channels.
        - out_channels (int): Number of output channels.
        - kernel_size (int): Size of the convolutional kernel (default is 3).
    """

    def __init__(self, in_channels, out_channels, kernel_size=3):
        super(Conv2dBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels,
                               kernel_size, padding=1)
        self.conv2 = nn.Conv2d(
            out_channels, out_channels, kernel_size, padding=1)
        self.activation = nn.ReLU()

    def forward(self, x):
        """
        Forward pass through the convolutional block.

        Args:
            - x (torch.Tensor): Input tensor.

        Returns:
            - torch.Tensor: Output tensor.
        """
        x = self.conv1(x)
        x = self.activation(x)
        x = self.conv2(x)
        x = self.activation(x)
        return x


class EncoderBlock(nn.Module):
    """
    Encoder block module.

    Args:
        - in_channels (int): Number of input channels.
        - out_channels (int): Number of output channels.
        - pool_size (tuple): Size of the pooling kernel (default is (2, 2)).
        - dropout (float): Dropout rate (default is 0.3).
    """

    def __init__(self, in_channels, out_channels, pool_size=(2, 2),
                 dropout=0.3):
        super(EncoderBlock, self).__init__()
        self.conv_block = Conv2dBlock(in_channels, out_channels)
        self.maxpool = nn.MaxPool2d(pool_size)
        self.dropout = nn.Dropout2d(dropout)

    def forward(self, x):
        """
        Forward pass through the encoder block.

        Args:
            - x (torch.Tensor): Input tensor.

        Returns:
            - torch.Tensor: Output tensor.
        """
        features = self.conv_block(x)
        pooled = self.maxpool(features)
        pooled = self.dropout(pooled)
        return features, pooled


class UNetEncoder(nn.Module):
    """
    Encoder network module.

    Args:
        - input_channels (int): Number of input channels.
    """

    def __init__(self, input_channels):
        super(UNetEncoder, self).__init__()
        self.block1 = EncoderBlock(input_channels, 64)
        self.block2 = EncoderBlock(64, 128)
        self.block3 = EncoderBlock(128, 256)
        self.block4 = EncoderBlock(256, 512)

    def forward(self, x):
        """
        Forward pass through the encoder network.

        Args:
            - x (torch.Tensor): Input tensor.

        Returns:
            - torch.Tensor: Output tensor.
        """
        f1, p1 = self.block1(x)
        f2, p2 = self.block2(p1)
        f3, p3 = self.block3(p2)
        f4, p4 = self.block4(p3)
        return p4, (f1, f2, f3, f4)


class Bottleneck(nn.Module):
    """
    Bottleneck module.

    """

    def __init__(self):
        super(Bottleneck, self).__init__()
        self.conv_block = Conv2dBlock(512, 1024)

    def forward(self, x):
        """
        Forward pass through the bottleneck module.

        Args:
            - x (torch.Tensor): Input tensor.

        Returns:
            - torch.Tensor: Output tensor.
        """
        return self.conv_block(x)


class DecoderBlock(nn.Module):
    """
    Decoder block module.

    Args:
        - in_channels (int): Number of input channels.
        - out_channels (int): Number of output channels.
        - kernel_size (int): Size of the convolutional kernel (default is 3).
        - stride (int): Stride size for the convolutional operation
            (default is 2).
        - dropout (float): Dropout rate (default is 0.3).
    """

    def __init__(self, in_channels, out_channels, kernel_size=3, stride=2,
                 dropout=0.3):
        super(DecoderBlock, self).__init__()
        self.deconv = nn.ConvTranspose2d(
            in_channels, out_channels, kernel_size, stride=stride, padding=1,
            output_padding=1)
        self.dropout = nn.Dropout2d(dropout)
        self.conv_block = Conv2dBlock(out_channels * 2, out_channels)

    def forward(self, x, conv_output):
        """
        Forward pass through the decoder block.

        Args:
            - x (torch.Tensor): Input tensor.
            - conv_output (torch.Tensor): Output tensor from the corresponding
            encoder block.

        Returns:
            - torch.Tensor: Output tensor.
        """
        x = self.deconv(x, output_size=conv_output.size())
        x = torch.cat([x, conv_output], dim=1)
        x = self.dropout(x)
        x = self.conv_block(x)
        return x


class UNetDecoder(nn.Module):
    """
    Decoder network module.

    Args:
        - output_channels (int): Number of output channels.
    """

    def __init__(self, output_channels):
        super(UNetDecoder, self).__init__()
        self.block6 = DecoderBlock(1024, 512)
        self.block7 = DecoderBlock(512, 256)
        self.block8 = DecoderBlock(256, 128)
        self.block9 = DecoderBlock(128, 64)
        self.final_conv = nn.Conv2d(64, output_channels, kernel_size=1)

    def forward(self, x, convs):
        """
        Forward pass through the decoder network.

        Args:
            - x (torch.Tensor): Input tensor.
            - convs (tuple): Tuple containing the output tensors from the
                encoder blocks.

        Returns:
            - torch.Tensor: Output tensor.
        """
        f1, f2, f3, f4 = convs
        x = self.block6(x, f4)
        x = self.block7(x, f3)
        x = self.block8(x, f2)
        x = self.block9(x, f1)
        outputs = self.final_conv(x)

        return outputs


class UNet(nn.Module):
    """
    UNet network module.

    Args:
        - input_channels (int): Number of input channels.
        - output_channels (int): Number of output channels.
    """

    def __init__(self, input_channels=3, output_channels=1, norm=True):
        super(UNet, self).__init__()
        self.encoder = UNetEncoder(input_channels)
        self.bottleneck = Bottleneck()
        self.decoder = UNetDecoder(output_channels)
        if norm:
            self.final_activation = torch.nn.Sigmoid()
        else:
            self.final_activation = torch.nn.ReLU()

    def forward(self, x):
        """
        Forward pass through the UNet network.

        Args:
            - x (torch.Tensor): Input tensor.

        Returns:
            - torch.Tensor: Output tensor.
        """
        encoder_output, convs = self.encoder(x)
        bottleneck_output = self.bottleneck(encoder_output)
        outputs = self.decoder(bottleneck_output, convs)
        return self.final_activation(outputs)
