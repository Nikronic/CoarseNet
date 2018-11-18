# -*- coding: utf-8 -*-
"""
Created on Sun Nov 18 12:49:08 2018

@author: Mohammad Doosti Lakhani
"""

# Importing Libraries
import torch
from torch import nn
import torch.nn.functional as f

class U_Net(nn.Module):
    def __init__(self, input_channels=1, output_channels=2, depth=5, number_of_filters=64):
        """
        Implementation of U-Net.
        Convolutional Networks for Biomedical Image Segmentation (Ronneberger et al., 2015)
        [https://arxiv.org/abs/1505.04597]
        
        Note: Default arguments are based on mentioned paper implementation.
        
        Args:
            **input_channels**: number of input channels of input images to network.
            
            **output_channels**: number of output channels of output images of network.
            
            **depth**: depth of network
            
            **number_of_filters**: number of filters in each layer (Each layer x2 the value).
            
        """
        super(U_Net, self).__init__()
        self.input_channels = input_channels
        self.output_channels = output_channels
        self.depth = depth
        self.number_of_filters = number_of_filters
        
        self.contracting_path = nn.ModuleList() # left side of shape of paper
        self.expansive_path = nn.ModuleList() # right side of shape of paper
    
    
    
        new_out_channels = self.number_of_filters
        new_in_channels = self.input_channels
        
        # Filling pathes with correspoding layers.
        for i in range(depth):
            self.contracting_path.append(Convolution(new_in_channels, new_out_channels))
            new_in_channels = new_out_channels
            new_out_channels = new_out_channels*2
        
        new_out_channels= new_out_channels 
        
        for i in reversed(range(depth-1)):
            new_out_channels = new_out_channels // 2
            self.expansive_path.append(UpConvolution(new_in_channels, new_out_channels))
            new_in_channels = new_out_channels
            
        self.last_layer = nn.Conv2d(new_in_channels, output_channels, kernel_size=1)   
        
        
    def forward(self, x):
        layers = []
        for i,downconv in enumerate(self.contracting_path):
            if i != len(self.contracting_path)-1: # add pooling to last layer
                x = downconv(x)
                layers.append(x)
                x = f.max_pool2d(x, kernel_size=2, stride=2)
        
        for i, upconv in enumerate(self.expansive_path):
            x = upconv(x, layers[-i-1])
        
        return self.last_layer(x)
            
            
        
class Convolution(nn.Module):
    def __init__(self, input_channel, output_channel):
        """
        This class provide two 3x3 convolutions (unpadded) each followed by a rectified linear unit (ReLU) which
        we used in both contracting and expanding phase.
        
        Args:
            **input_channel**: Input size of layer
            
            **output_channel**: Output size of layer
        
        """
        
        super(Convolution, self).__init__()
        
        layer = []
        layer.append(nn.Conv2d(input_channel, output_channel, kernel_size=3))
        layer.append(nn.ReLU())
        
        layer.append(nn.Conv2d(output_channel, output_channel, kernel_size=3))
        layer.append(nn.ReLU())
        
        self.layer = nn.Sequential(*layer) # hold the order
        
    def forward(self, x):
        return self.layer(x)
        
    
    
class UpConvolution(nn.Module):
    def __init__(self, input_channel, output_channel):
        """
        A 2x2 convolution which concatenate with cropped features from contracting phase, and two 3x3 convolution,
        each followed by a rectified linear unit (ReLU) (our conolution class).
        
        Args:
            **input_channel**: Input size of layer
            
            **output_channel**: Output size of layer
        
        """
        
        super(UpConvolution, self).__init__()
        
        
        self.top_layer = nn.ConvTranspose2d(input_channel, output_channel, kernel_size=2)
        self.conv = Convolution(input_channel, output_channel)
        
    def crop(self, layer, target_size):
        _, _, layer_height, layer_width = layer.size()
        diff_y = (layer_height - target_size[0]) // 2
        diff_x = (layer_width - target_size[1]) // 2
        return layer[:, :, diff_y:(diff_y + target_size[0]), diff_x:(diff_x + target_size[1])]
    
    def forward(self, x, layer_map):
        layer = self.top_layer(x)
        layer = self.conv(layer = torch.cat([layer, self.crop(layer_map, layer.shape[2:])], 1))
        return layer
    
    

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = U_Net().to(device)
optim = torch.optim.Adam(model.parameters())

x = torch.randn(1, 1, 320, 320)

zzz = model(x)
zzz2 = model(x)
