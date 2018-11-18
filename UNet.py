# -*- coding: utf-8 -*-
"""
Created on Sun Nov 18 17:32:42 2018

@author: Mohammad Doosti Lakhani
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class DoubleConvolution(nn.Module):
  def __init__(self, input_channel, output_channel):
    """
    It consists of the repeated
    application of two 3x3 convolutions (unpadded convolutions), each followed by
    a recti
ed linear unit (ReLU)
    
    Args:
      
      **input_channel**: input channel size
      
      **output_channel**: output channel size
    """
    
    
    super(DoubleConvolution, self).__init__()
    layers = []
    layers.append(nn.Conv2d(input_channel, output_channel, kernel_size=3, stride=1))
    layers.append(nn.ReLU())
    layers.append(nn.Conv2d(output_channel, output_channel, kernel_size = 3, stride=1))
    layers.append(nn.ReLU())
   
    self.layers = nn.Sequential(*layers)

  def forward(self, x):
    return self.layers(x)
       

class Contract(nn.Module):
  def __init__(self, input_channel, output_channel):
    """
    It consists of a DoubleConvolution followed by a 2x2 MaxPooling operation with stride 2 for downsampling.
    
    Args:
      
      **input_channel**: input channel size
      
      **output_channel**: output channel size
      
    """
    super(Contract, self).__init__()
    layers = []
    layers.append(nn.MaxPool2d(kernel_size=2, stride=2))
    layers.append(DoubleConvolution(input_channel, output_channel))
    self.layers = nn.Sequential(*layers)
    
  def forward(self, x):
    return self.layers(x)
    
class Expand(nn.Module):
  def __init__(self, input_channel, output_channel):
    """
    This path consists of an upsampling of the feature map followed by a 
    2x2 convolution ("up-convolution" or Transformed Convolution) that halves the number of 
    feature channels, a concatnation with the correspondingly cropped feature map from Cantract phase and
    a DoubleConvolution 
    
    Args:
      
      **input_channel**: input channel size
      
      **output_channel**: output channel size
    
    """
    super(Expand, self).__init__()
    self.up_conv = nn.ConvTranspose2d(input_channel//2, output_channel//2, kernel_size=2, stride=2)
    self.layers = DoubleConvolution(input_channel, output_channel)
    

  def forward(self, x1, x2):
    x1 = self.up_conv(x1)
    delta_x = x1.size()[2] - x2.size()[2]
    delta_y = x1.size()[3] - x2.size()[3]
    x2 = F.pad(x1, pad=(delta_x//2, delta_y//2) , mode='constant', value=0)
    x = torch.cat(seq = (x2, x1), dim=1)
    x = self.layers(x)
    return x
  
    
class FinalConvolution(nn.Module):
  def __init__(self, input_channel, output_channel):
    """
    At the final layer, a 1x1 convolution is used to map each 64-component feature vector to the desired
    number of classes.
    
    Args:
      
      **input_channel**: input channel size
      
      **output_channel**: output channel size
      
    """
    super(FinalConvolution, self).__init__()
    self.layer =  nn.Conv2d(input_channel, output_channel, kernel_size=1)
    
  def forward(self, x):
    return self.layer(x)
    
    
class UNet(nn.Module):
  def __init__(self, input_channels=1, output_channels=2, depth=5, filters=64):
    """
    Implementation of U-Net.
      Convolutional Networks for Biomedical Image Segmentation (Ronneberger et al., 2015)
      [https://arxiv.org/abs/1505.04597]
      
      Note: Default arguments are based on mentioned paper implementation.
      
      Args:
        **input_channels**: number of input channels of input images to network.
        
        **output_channels**: number of output channels of output images of network.
        
        **depth**: depth of network
        
        **filters**: number of filters in each layer (Each layer x2 the value).
          
    """
    
    super(UNet, self).__init__()
    self.input_channels = input_channels
    self.output_channels = output_channels
    self.depth = depth
    self.filters = filters
    
    self.contracting_path = nn.ModuleList() # left side of shape of network in the paper
    self.expansive_path = nn.ModuleList() # right side of shape of network in the paper
    
    prev_channels = self.input_channels
    
    self.contracting_path.append(DoubleConvolution(prev_channels, filters))
    prev_channels = filters
    filters *=2
    for _ in range(depth-1):
      self.contracting_path.append(Contract(prev_channels, filters))
      prev_channels = filters
      filters *= 2
     
    filters = prev_channels//2
    for _ in reversed(range(depth-1)):
      self.expansive_path.append(Expand(prev_channels, filters))
      prev_channels = filters
      filters //= 2
    
    self.final = FinalConvolution(prev_channels, output_channels)
  
  def forward(self, x):
    layers = []
    for _, l in enumerate(self.contracting_path):
      layers.append(l(x))
    
    up = self.expanding_path[0]
    x = up(layers[-1], layers[-2])
    for i, l in enumerate(self.expanding_path):
      if i == 0:
        pass
      else:
        x = l(x, layers[-i-2])
    x = self.final(x)
    return F.Sigmoid(x)
    
    
  
  
from torch.autograd import Variable
import numpy as np
model = UNet()
x = Variable(torch.FloatTensor(np.random.random((1, 3, 320, 320))))
x = torch.randn(1, 1, 320, 320)
out = model(x)
