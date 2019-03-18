# %% Import libraries
import torch
import torch.nn as nn
import torch.nn.functional as F


# %% Submodules
class CL(nn.Module):
    def __init__(self, input_channel, output_channel):
        """
        It consists of the 4x4 convolutions with stride=2, padding=1, each followed by
        a leaky rectified linear unit (Leaky ReLU)

        :param input_channel: input channel size
        :param output_channel: output channel size
        """

        assert (input_channel > 0 and output_channel > 0)

        super(CL, self).__init__()
        layers = [nn.Conv2d(input_channel, output_channel, kernel_size=4, stride=2, padding=1), nn.LeakyReLU(0.2)]
        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        return self.layers(x)


# %%
class CBL(nn.Module):
    def __init__(self, input_channel, output_channel):
        """
        It consists of the 4x4 convolutions with stride=2, padding=1, and a batch normalization, followed by
        a leaky rectified linear unit (ReLU)

        :param input_channel: input channel size
        :param output_channel: output channel size
        """
        assert (input_channel > 0 and output_channel > 0)

        super(CBL, self).__init__()
        layers = [nn.Conv2d(input_channel, output_channel, kernel_size=4, stride=2, padding=1),
                  nn.BatchNorm2d(num_features=output_channel), nn.LeakyReLU(0.2)]
        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        return self.layers(x)


# %%
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
        layers = [nn.ConvTranspose2d(input_channel, output_channel, kernel_size=ks, stride=s, padding=1),
                  nn.ELU(alpha=1)]
        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        return self.layers(x)


# %%
class Contract(nn.Module):
    def __init__(self, input_channel, output_channel, module='cbl'):
        """
        It consists of a CL or CBL followed by a 2x2 MaxPooling operation with stride 2 for down sampling.


        :param input_channel: input channel size
        :param output_channel: output channel size
        :param module: using Convolution->ReLU (CL class) or Convolution->BathNorm->ReLU (CBL class)
                Convolution->ELU (CE class) for first layer of Expand (decoder) path
        """

        assert (input_channel > 0 and output_channel > 0)

        super(Contract, self).__init__()

        layers = []
        if module == 'cl':
            layers.append(CL(input_channel, output_channel))
        elif module == 'ce':
            layers.append(CE(input_channel, output_channel))
        else:
            layers.append(CBL(input_channel, output_channel))

        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        return self.layers(x)


# %%
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
        self.layers = CE(input_channel * 2, output_channel, ks, s)

    def forward(self, x1, x2):
        delta_x = x1.size()[2] - x2.size()[2]
        delta_y = x1.size()[3] - x2.size()[3]
        x2 = F.pad(x2, pad=(delta_x // 2, delta_y // 2, delta_x // 2, delta_y // 2), mode='constant', value=0)
        x = torch.cat((x2, x1), dim=1)
        x = self.layers(x)
        return x


# %%
class C(nn.Module):
    def __init__(self, input_channel, output_channel):
        """
        At the final layer, a 3x3 convolution is used to map each 64-component feature vector to the desired
        number of classes.

        :param input_channel: input channel size
        :param output_channel: output channel size
        """
        super(C, self).__init__()
        self.layer = nn.Conv2d(input_channel, output_channel, kernel_size=3, padding=1, stride=1)

    def forward(self, x):
        return self.layer(x)


#%% Main CLass
class CoarseNet(nn.Module):
    def __init__(self, input_channels=3, output_channels=3):
        """
        Implementation of CoarseNet, a modified version of UNet.
        (https://arxiv.org/abs/1505.04597 - Convolutional Networks for Biomedical Image Segmentation (Ronneberger et al., 2015))

        :param input_channels: number of input channels of input images to network.
        :param output_channels: number of output channels of output images of network.
        """

        super(CoarseNet, self).__init__()
        self.input_channels = input_channels
        self.output_channels = output_channels

        # Encoder
        self.cl0 = Contract(input_channels, 64, module='cl')
        self.cbl0 = Contract(64, 128)
        self.cbl1 = Contract(128, 256)
        self.cbl2 = Contract(256, 512)
        self.cl1 = Contract(512, 512, module='cl')

        # Decoder
        self.ce0 = Contract(512, 512, module='ce')
        self.ce1 = Expand(512, 256)
        self.ce2 = Expand(256, 128)
        self.ce3 = Expand(128, 64)
        self.ce4 = Expand(64, 64)
        self.ce5 = CE(64, 64, ks=3, s=1)

        # final
        self.final = C(64, self.output_channels)

    def forward(self, x):
        out = self.cl0(x)  # 3>64
        out2 = self.cbl0(out)  # 64>128
        out3 = self.cbl1(out2)  # 128>256
        out4 = self.cbl2(out3)  # 256>512
        out5 = self.cl1(out4)  # 512>512
        in0 = self.ce0(out5)

        in1 = self.ce1(out4, in0)  # 512>512
        in2 = self.ce2(out3, in1)  # 512>256
        in3 = self.ce3(out2, in2)  # 256>128
        in4 = self.ce4(out, in3)  # 128>64
        f = self.ce5(in4)
        f = self.final(f)
        return f


# %% tests
# z = torch.randn(1, 3, 256, 256)
# model = CoarseNet()
# o = model(z)
