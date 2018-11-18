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
    layers.add(nn.Conv2d(input_channel, output_channel, kernel_size=3, stride=1))
    layers.add(nn.ReLU())
    layers.add(nn.Conv2d(output_channel, output_channel, kernel_size = 3, stride=1))
    layers.add(nn.ReLU())
   
    self.layers = nn.Sequential(*layers)

    def forward(self, x):
      return self.layers[x]
       

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
    layers.append(nn.MaxPool2d(stride=2))
    layers.append(DoubleConvolution(input_channel, output_channel))
    self.layers = nn.Sequential(*layers)
    
    def forward(self, x):
      return self.layers[x]
    
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
    
    

