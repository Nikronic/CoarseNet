# %% libraries
import torch.nn as nn
import torch
from vgg import vgg16_bn


class CoarseLoss(nn.Module):
    def __init__(self, w1=50, w2=1):
        """
        A weighted sum of pixel-wise L1 loss and sum of L2 loss of Gram matrices.

        :param w1: weight of L1  (pixel-wise)
        :param w2: weight of L2 loss (Gram matrix)
        """
        super(CoarseLoss, self).__init__()
        self.w1 = w1
        self.w2 = w2
        self.l1 = nn.L1Loss(reduction='mean')
        self.l2 = nn.MSELoss(reduction='mean')
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

    @staticmethod
    def patch_feature(self):
        raise NotImplementedError('This method has not been implemented yet.')

    def forward(self, y, y_pred):
        # TODO do patching of y and y_pred

        y_patch_pool2 = self.patch_feature(self.vgg16_bn(y[0]))
        y_patch_pool5 = self.patch_feature(self.vgg16_bn(y[1]))
        y_pred_patch_pool2 = self.patch_feature(self.vgg16_bn(y_pred[0]))
        y_pred_patch_pool5 = self.patch_feature(self.vgg16_bn(y_pred[1]))

        loss = self.w1 * self.l1(y, y_pred) + \
               self.w2 * (self.l2(self.gram_matrix(y_patch_pool2), self.gram_matrix(y_pred_patch_pool2)) +
                          self.l2(self.gram_matrix(y_patch_pool5), self.gram_matrix(y_pred_patch_pool5)))

        return loss
