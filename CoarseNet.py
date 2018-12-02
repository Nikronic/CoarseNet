# -*- coding: utf-8 -*-
"""
Created on Sun Nov 18 16:54:05 2018

@author: Mohammad Doosti Lakhani
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class CL(nn.Module):
    def __init__(self, input_channel, output_channel):
        """
        It consists of the 4x4 convolutions with stride=2, padding=1, each followed by
        a rectified linear unit (ReLU)

        :param input_channel: input channel size
        :param output_channel: output channel size
        """

        assert (input_channel > 0 and output_channel > 0)

        super(CL, self).__init__()
        layers = []
        layers.append(nn.Conv2d(input_channel, output_channel, kernel_size=4, stride=2, padding=1))
        layers.append(nn.ReLU())

        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        return self.layers(x)


class CBL(nn.Module):
    def __init__(self, input_channel, output_channel):
        """
        It consists of the 4x4 convolutions with stride=2, padding=1, and a batch normalization, followed by
        a rectified linear unit (ReLU)

        :param input_channel: input channel size
        :param output_channel: output channel size
        """
        assert (input_channel > 0 and output_channel > 0)

        super(CBL, self).__init__()
        layers = []
        layers.append(nn.Conv2d(input_channel, output_channel, kernel_size=4, stride=2, padding=1))
        layers.append(nn.BatchNorm2d(num_features=output_channel))
        layers.append(nn.ReLU())

        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        return self.layers(x)


class CE(nn.Module):
    def __init__(self, input_channel, output_channel, ks=4, s=2):
        """
        It consists of the 4x4 convolutions with stride=2, padding=1, each followed by
        a exponential linear unit (ELU)

        :param input_channel: input channel size
        :param output_channel: output channel size
        :param ks: kernel size
        :param s: stride size
        """
        assert (input_channel > 0 and output_channel > 0)

        super(CE, self).__init__()
        layers = []
        layers.append(nn.Conv2d(input_channel, output_channel, kernel_size=ks, stride=s, padding=1))
        layers.append(nn.ELU(alpha=1))

        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        return self.layers(x)


class Contract(nn.Module):
    def __init__(self, input_channel, output_channel, is_cl=False):
        """
        It consists of a CL or CBL followed by a 2x2 MaxPooling operation with stride 2 for down sampling.


        :param input_channel: input channel size
        :param output_channel: output channel size
        :param is_cl: using Convolution->ReLU (CL class) or Convolution->BathNorm->ReLU (CBL class)
        """

        assert (input_channel > 0 and output_channel > 0)

        super(Contract, self).__init__()

        layers = []
        if is_cl:
            layers.append(CL(input_channel, output_channel))
        else:
            layers.append(CBL(input_channel, output_channel))
        layers.append(nn.MaxPool2d(kernel_size=2, stride=2))

        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        return self.layers(x)


class Expand(nn.Module):
    def __init__(self, input_channel, output_channel, ks=4, s=2):
        """
        This path consists of an up sampling of the feature map followed by a
        4x4 convolution ("up-convolution" or Transformed Convolution) that halves the number of
        feature channels, a concatenation with the correspondingly cropped feature map from Contract phase


        :param input_channel: input channel size
        :param output_channel: output channel size
        """
        super(Expand, self).__init__()

        self.up_conv = nn.ConvTranspose2d(input_channel, output_channel, kernel_size=2, stride=2, padding=1)
        self.layers = CE(input_channel, output_channel, ks, s)

    def forward(self, x1, x2):
        x1 = self.up_conv(x1)
        delta_x = x1.size()[2] - x2.size()[2]
        delta_y = x1.size()[3] - x2.size()[3]
        x2 = F.pad(x2, pad=(delta_x // 2, delta_y // 2, delta_x // 2, delta_y // 2), mode='constant', value=0)
        x12 = torch.cat((x2, x1), dim=1)
        x = self.layers(x12)
        return x


class C(nn.Module):
    def __init__(self, input_channel, output_channel):
        """
        At the final layer, a 3x3 convolution is used to map each 64-component feature vector to the desired
        number of classes.

        :param input_channel**: input channel size
        :param output_channel**: output channel size
        """
        super(C, self).__init__()
        self.layer = nn.Conv2d(input_channel, output_channel, kernel_size=3, padding=1)

    def forward(self, x):
        return self.layer(x)


class CoarseNet(nn.Module):
    def __init__(self, input_channels=3, output_channels=3, depth=5, filters=64):
        """
        Implementation of CoarseNet, a modified version of UNet. (Part of )TODO add paper citation

        :param input_channels: number of input channels of input images to network.
        :param output_channels: number of output channels of output images of network.
        :param depth: depth of network
        :param filters: number of filters in each layer (Each layer x2 the value).
        """

        super(CoarseNet, self).__init__()
        self.input_channels = input_channels
        self.output_channels = output_channels
        self.depth = depth
        self.filters = filters

        self.contracting_path = nn.ModuleList()  # left side of shape of network in the paper
        self.expansive_path = nn.ModuleList()  # right side of shape of network in the paper

        prev_channels = self.input_channels

        prev_channels = input_channels
        for i in range(depth):
            if i==0:
                self.contracting_path.append(Contract(prev_channels, filters, is_cl=True))
            elif i==depth-1:
                self.contracting_path.append(Contract(prev_channels, filters, is_cl=True))
            else:
                self.contracting_path.append(Contract(prev_channels, filters))
            prev_channels = filters
            filters *= 2

        filters = prev_channels // 2
        for i in reversed(range(depth + 1)):
            if i == 1:
                self.expansive_path.append(Expand(prev_channels, prev_channels))
            elif i == 0:
                self.expansive_path.append(Expand(prev_channels, filters, ks=3, s=1))
            else:
                self.expansive_path.append(Expand(prev_channels, filters))
            prev_channels = filters
            filters //= 2

        self.final = C(prev_channels, output_channels)

    def forward(self, x):
        layers = []
        for i, l in enumerate(self.contracting_path):
            if i == 0:
                layers.append(l(x))
            else:
                x = layers[i - 1]
                layers.append(l(x))

        up = self.expansive_path[0]
        x = up(layers[-1], layers[-2])
        for i, l in enumerate(self.expansive_path):
            if i == 0:
                pass
            else:
                x = l(x, layers[-i - 2])
        x = final(x)
        return x


x = torch.randn(1, 3, 512, 512) 
model = CoarseNet()
o = model(x)

model = Contract(3, 64, is_cl=True)
out = model(x)
model = Contract(64, 128)
out2 = model(out)
model = Contract(128, 256)
out3 = model(out2)
model = Contract(256, 512)
out4 = model(out3)
model = Contract(512, 512, is_cl=True)
out5 = model(out4)

model = Expand(1024, 512)
in1 = model(out5, out4)
model = Expand(512, 256)
in2 = model(in1, out3)
model = Expand(256, 128)
in3 = model(in2, out2)
model = Expand(128, 64)
in4 = model(in3, out)
model = C(64, 3)
final = model(in4)
