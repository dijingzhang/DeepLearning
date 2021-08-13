import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Parameter
import math


# ArcFace
class ArcMarginProduct(nn.Module):
    """
    Implement of large margin arc distance (ArcFace)
    Args:
        in_features: size of each input sample
        out_features: size of each output sample
        s: norm of input feature
        m: margin

        cos(theta+m)
    """

    def __init__(self, in_features, out_features, s=30.0, m = 0.50, easy_margin=False):
        super(ArcMarginProduct, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.s = s
        self.m = m
        self.weight = Parameter(torch.FloatTensor(out_features, in_features))
        nn.init.xavier_normal_(self.weight)

        self.easy_margin = easy_margin
        self.cos_m = math.cos(m)
        self.sin_m = math.sin(m)
        self.th = math.cos(math.pi - m)
        self.mm = math.sin(math.pi - m) * m

    def forward(self, input, label):
        # ---------------------cos(theta) &  phi(theta) ----------------------
        # torch.nn.functional.linear(input, weight, bias=None)
        # y = x * W^T + b
        cosine = F.linear(F.normalize(input), F.normalize((self.weight)))
        sine = torch.sqrt(1.0 - torch.pow(cosine, 2))
        # cos(a + b) = cos(a) * cos(b) - sine(a) * sine(b)
        phi = cosine * self.cos_m - sine * self.sin_m
        if self.easy_margin:
            # torch.where(condition, x, y) -> Tensor
            # condition(ByteTensor) - When True (nonzero), yield x, otherwise yield y
            # x (Tensor) - values selected at indices where condition is True
            # y (Tensor) - values selected at indices where condition is False
            # return:
            # A tensor of shape equal to the broadcasted shape of condition, x, y
            # cosine > 0 means two class is similar, thus use the phi which make it
            phi = torch.where(cosine > 0, phi, cosine)
        else:
            phi = torch.where(cosine > self.th, phi, cosine - self.mm)
        # --------------------- convert label to one-hot ----------------------
        # one-hot = torch.zeros(cosine.size(), requires_grad=True, device='cuda')
        # update cos(theta+m) to tensor
        one_hot = torch.zeros_like(cosine, device='cuda')
        # scatter_(dim, index, src)
        one_hot.scatter_(1, label.view(-1, 1).long(), 1)
        # --------------------- torch.where(out_i = x_i if condition_i else y_i) ----------------------
        output = (one_hot * phi) + ((1.0 - one_hot) * cosine)
        output *= self.s

        return output