# %% libraries
import torch.nn as nn
import torch
from vgg import vgg16_bn
import numpy as np


class CoarseLoss(nn.Module):
    def __init__(self, w1=50, w2=1, weight_vgg=None):
        """
        A weighted sum of pixel-wise L1 loss and sum of L2 loss of Gram matrices.

        :param w1: weight of L1  (pixel-wise)
        :param w2: weight of L2 loss (Gram matrix)
        :param weight_vgg: weight of VGG extracted features (should be add up to 1.0)
        """
        super(CoarseLoss, self).__init__()
        if weight_vgg is None:
            weight_vgg = [0.5, 0.5, 0.5, 0.5, 0.5]
        self.w1 = w1
        self.w2 = w2
        self.l1 = nn.L1Loss(reduction='mean')
        self.l2 = nn.MSELoss(reduction='mean')
        self.weight_vgg = weight_vgg
        self.vgg16_bn = vgg16_bn(pretrained=True)

    # reference: https://github.com/pytorch/tutorials/blob/master/advanced_source/neural_style_tutorial.py
    @staticmethod
    def gram_matrix(mat):
        """
        Return Gram matrix

        :param mat: A matrix  (a=batch size(=1), b=number of feature maps,
        (c,d)=dimensions of a f. map (N=c*d))
        :return: Normalized Gram matrix
        """
        a, b, c, d = mat.size()
        features = mat.view(a * b, c * d)
        gram = torch.mm(features, features.t())
        return gram.div(a * b * c * d)

    def forward(self, y, y_pred):
        y_vgg = self.vgg16_bn(y)
        y_pred_vgg = self.vgg16_bn(y_pred)
        loss_vgg = [self.l2(self.gram_matrix(ly), self.gram_matrix(lp)) for ly, lp in zip(y_vgg, y_pred_vgg)]

        loss = self.w1 * self.l1(y, y_pred) + \
               self.w2 * np.dot(loss_vgg, self.weight_vgg)
        return loss
