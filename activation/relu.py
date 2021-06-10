class ReLU(Activation):

    """
    ReLU non-linearity
    """

    def __init__(self):
        super(ReLU, self).__init__()

    def forward(self, x):
        self.state = np.where(x > 0, x, 0)
        return self.state

    def derivative(self):
        copy = np.copy(self.state)
        deriv = np.where(copy > 0.0, 1.0, 0.0)
        return deriv