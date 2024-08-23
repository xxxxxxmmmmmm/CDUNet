import numpy as np
import torch.nn as nn
import torch
from torch.nn.modules.loss import _Loss
import torch.nn.functional as F

class SobelOperator(nn.Module):
    def __init__(self, epsilon=1e-4):
        super().__init__()
        self.epsilon = epsilon

        self.register_buffer('conv_x', torch.Tensor([[1, 0, -1], [2, 0, -2], [1, 0, -1]])[None, None, :, :] / 4)
        self.register_buffer('conv_y', torch.Tensor([[1, 2, 1], [0, 0, 0], [-1, -2, -1]])[None, None, :, :] / 4)

    def forward(self, x):
        grad_x = F.conv2d(x, self.conv_x, bias=None, stride=1, padding=1)
        grad_y = F.conv2d(x, self.conv_y, bias=None, stride=1, padding=1)
        x = torch.sqrt(grad_x ** 2 + grad_y ** 2 + self.epsilon)
        return x


class GradLoss(nn.Module):
    def __init__(self):
        super(GradLoss, self).__init__()
        self.sobel = SobelOperator(1e-4)

    def forward(self, pr, gt):
        gt_sobel = self.sobel(gt)
        pr_sobel = self.sobel(pr)
        grad_loss = F.l1_loss(gt_sobel, pr_sobel)
        return grad_loss

# class mixloss(_Loss):
#     def __init__(self,w1 = 1,w2 = 0,w3 = 0.1):
#         super(mixloss,self).__init__()
#         self.w1 = w1
#         self.w2 = w2
#         self.w3 = w3
#         self.criterion = nn.L1Loss()
#         self.g = GradLoss()
#
#     def forward(self,pred,y,pred2,y2):
#         loss1 = self.criterion(pred,y)
#         loss2 = self.criterion(pred2,y2)
#         self.g.to('cuda')
#         loss3 = self.g(pred,y)
#         #print(loss1)
#         #print(loss2)
#         #print(loss3)
#         return self.w1*loss1 + self.w2*loss2 + self.w3*loss3

class mixloss(_Loss):
    def __init__(self,w1 = 0.95,w2 = 0.05):
        super(mixloss,self).__init__()
        self.w1 = w1
        self.w2 = w2
        self.criterion = nn.L1Loss()
        self.g = GradLoss()

    def forward(self,pred,y):
        loss1 = self.criterion(pred,y)
        self.g.to('cuda')
        loss2 = self.g(pred,y)
        # print(loss1)
        # print(loss2)
        return self.w1*loss1 + self.w2*loss2
