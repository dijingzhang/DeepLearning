import torch
import torch.nn as nn
from torch.autograd import Variable
import numpy as np

class LabelSmoothing(nn.Module):
    def __init__(self, ncls, smoothing=0.0):
        super(LabelSmoothing, self).__init__()

        self.criterion = nn.CrossEntropyLoss(size_average=False)
        self.smoothing = smoothing
        self.confidence = 1.0 - smoothing  # if i = y
        self.ncls = ncls
        self.true_dist = None

    def forward(self, x, target):
        assert x.size(1) == self.ncls
        true_dist = x.data.clone()
        true_dist.fill_(self.smoothing / (self.ncls - 1))  # otherwise
        true_dist.scatter_(1, target.data.unsqueeze(1), self.confidence)

        return self.criterion(x, true_dist)