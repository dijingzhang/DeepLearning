import numpy as np
from base import Activation

class LeakyReLU(Activation):

    """
    LeakyReLU non-linearity
    LeakyReLU = { x if x >= 0
            { a * x otherwise
            where a is a negative slope (fixed) (default = 1e-2)
    """

    def __init__(self, a):
        super(LeakyReLU, self).__init__()
        self.a = a

    def forward(self, x):
        self.state = np.where(x > 0, x, self.a * x)
        return self.state

    def derivative(self):
        copy = np.copy(self.state)
        deriv = np.where(copy > 0.0, 1.0, self.a)
        return deriv