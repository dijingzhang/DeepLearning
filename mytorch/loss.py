import numpy as np
import os

# The following Criterion class will be used again as the basis for a number
# of loss functions (which are in the form of classes so that they can be
# exchanged easily (it's how PyTorch and other ML libraries do it))

class Criterion(object):
    """
    Interface for loss functions.
    """

    # Nothing needs done to this class, it's used by the following Criterion classes

    def __init__(self):
        self.logits = None
        self.labels = None
        self.loss = None

    def __call__(self, x, y):
        return self.forward(x, y)

    def forward(self, x, y):
        raise NotImplemented

    def derivative(self):
        raise NotImplemented

class SoftmaxCrossEntropy(Criterion):
    """
    Softmax loss
    """

    def __init__(self):
        super(SoftmaxCrossEntropy, self).__init__()

    def forward(self, x, y):
        """
        Argument:
            x (np.array): (batch size, 10)
            y (np.array): (batch size, 10)
        Return:
            out (np.array): (batch size, )
        """
        self.logits = x
        self.labels = y
        max = np.max(x, axis=1, keepdims=True)
        x_normalized = x - max
        sum = np.sum(np.exp(x_normalized), axis=1, keepdims=True)
        res = x_normalized - np.log(sum)
        loss = -np.sum(y * res, axis=1)
        self.loss = loss
        return self.loss


    def derivative(self):
        """
        Return:
            out (np.array): (batch size, 10)
        """

        x = self.logits
        y = self.labels
        max = np.max(x, axis=1, keepdims=True)
        x_normalized = np.exp(x - max)
        sum = np.sum(x_normalized, axis=1, keepdims=True)
        res = x_normalized / sum
        deriv_loss = (res - y)
        return deriv_loss


class L2Loss(Criterion):
    """
    Euclidean distance loss
    """

    def __init__(self):
        super(L2Loss, self).__init__()

    def forward(self, x, y):
        """
        Argument:
            x (np.array): (batch size, 10)
            y (np.array): (batch size, 10)
        Return:
            out (np.array): (batch size, )
        """
        self.logits = x
        self.labels = y

        loss = np.sum((x - y) ** 2, axis=1) / x.shape[1]
        self.loss = loss
        return self.loss


    def derivative(self):
        """
        Return:
            out (np.array): (batch size, 10)
        """

        x = self.logits
        y = self.labels
        n = x.shape[1]
        deriv_loss = (2 / n) * (x - y)
        return deriv_loss
