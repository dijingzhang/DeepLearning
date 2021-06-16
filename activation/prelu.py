import numpy as np
from base import Activation

class PReLU(Activation):

    """
    PReLU non-linearity
    PReLU(x) = max(0, x) + a * min(0, x)
    PReLU = { x if x >= 0
            { a * x otherwise
            where a is a learnable parameter, initialized as 0.25
    More detail please refer to:
    https://openaccess.thecvf.com/content_iccv_2015/papers/He_Delving_Deep_into_ICCV_2015_paper.pdf
    """

    def __init__(self, mu, sigma, num_parameters=1, init=0.25):
        """
        num_parameters (int) – number of a to learn. It should be 1 or the number of channels at input. Default: 1
        init (float) – the initial value of aa. Default: 0.25
        """
        super(PReLU, self).__init__()
        self.a = np.ones(num_parameters) * init
        self.momentum_a = np.zeros_like(self.a)
        self.mu = mu  # parameter for momentum
        self.sigma = sigma  # learning rate

    def forward(self, x):
        """
        x (np.array)  [bs, in_channel, height, width]
        """
        self.state = np.max(0, x) + self.a * np.min(0, x)
        return self.state

    def derivative(self, delta):
        copy = np.copy(self.state)
        deriv = np.where(copy > 0.0, 1.0, self.a)
        self.momentum_a = self.momentum_a * self.mu + self.sigma * np.sum(np.min(np.zeros_like(copy), copy) * delta, axis=(0, 2, 3))
        self.a -= self.momentum_a
        return deriv