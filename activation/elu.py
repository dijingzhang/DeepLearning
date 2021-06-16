import numpy as np
from base import Activation

class ELU(Activation):

    """
    ELU non-linearity
    ELU(x) = max(0, x) + a * min(0, x)
    ELU = { x if x >= 0
          { a * (exp(x)-1) otherwise (default a is 5)

    """

    def __init__(self, a):
        super(ELU, self).__init__()
        self.a = a

    def forward(self, x):
        self.x = x
        self.state = np.where(x > 0, x, self.a * (np.exp(x) - 1))
        return self.state

    def derivative(self):
        copy = np.copy(self.state)
        deriv = np.where(copy > 0.0, 1.0, self.a * np.exp(self.x))
        return deriv