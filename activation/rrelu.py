import numpy as np
from base import Activation

class RReLU(Activation):

    """
    RReLU non-linearity
    RReLU = { x if x >= 0
            { a * x otherwise
            where a is randomly sampled from uniform distribution u(lower, upper)
    """

    def __init__(self, upper, lower):
        super(RReLU, self).__init__()
        self.upper = upper
        self.lower = lower

    def forward(self, x):
        self.a = np.random.uniform(self.upper, self.lower)
        self.state = np.where(x > 0, x, self.a * x)
        return self.state

    def derivative(self):
        copy = np.copy(self.state)
        deriv = np.where(copy > 0.0, 1.0, self.a)
        return deriv