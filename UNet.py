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
       
       
       