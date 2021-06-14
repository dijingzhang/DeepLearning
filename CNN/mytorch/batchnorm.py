# Do not import any additional 3rd party external libraries as they will not
# be available to AutoLab and are not needed (or allowed)

import numpy as np

class BatchNorm(object):

    def __init__(self, in_feature, alpha=0.9):

        # You shouldn't need to edit anything in init

        self.alpha = alpha
        self.eps = 1e-8
        self.x = None
        self.norm = None
        self.out = None

        # The following attributes will be tested
        self.var = np.ones((1, in_feature))
        self.mean = np.zeros((1, in_feature))

        self.gamma = np.ones((1, in_feature))
        self.dgamma = np.zeros((1, in_feature))

        self.beta = np.zeros((1, in_feature))
        self.dbeta = np.zeros((1, in_feature))

        # inference parameters
        self.running_mean = np.zeros((1, in_feature))
        self.running_var = np.ones((1, in_feature))

    def __call__(self, x, eval=False):
        return self.forward(x, eval)

    def forward(self, x, eval=False):
        """
        Argument:
            x (np.array): (batch_size, in_feature)
            eval (bool): inference status

        Return:
            out (np.array): (batch_size, in_feature)

        NOTE: The eval parameter is to indicate whether we are in the
        training phase of the problem or are we in the inference phase.
        So see what values you need to recompute when eval is True.
        """

        if eval == True:
            self.out = self.gamma * ((x - self.running_mean) / np.power((self.running_var + self.eps), 0.5)) + self.beta
        else:
            self.x = x

            self.mean = np.mean(x, axis=0, keepdims=True)
            self.var = np.var(x, axis=0, keepdims=True)
            self.norm = (x - self.mean) / np.power((self.var + self.eps), 0.5)
            self.out = self.gamma * self.norm + self.beta

            # Update running batch statistics
            self.running_mean = self.alpha * self.running_mean + (1 - self.alpha) * self.mean
            self.running_var = self.alpha * self.running_var + (1 - self.alpha) * self.var

        return self.out


    def backward(self, delta):
        """
        Argument:
            delta (np.array): (batch size, in feature)
        Return:
            out (np.array): (batch size, in feature)
        """
        m = delta.shape[0]

        self.dgamma = np.sum(delta * self.norm, axis=0, keepdims=True)
        self.dbeta = np.sum(delta, axis=0, keepdims=True)

        dnorm = delta * self.gamma
        dvar = -0.5 * np.sum(dnorm * (self.x-self.mean) * np.power((self.var + self.eps), -1.5), axis=0, keepdims=True)
        dmean = -np.sum(dnorm * np.power((self.var + self.eps), -0.5), axis=0, keepdims=True) -  \
                (2 / m) * dvar * np.sum((self.x-self.mean), axis=0, keepdims=True)
        dx = dnorm * np.power((self.var+self.eps), -0.5) + dvar * (self.x - self.mean) * (2 / m) + (1 / m) * dmean
        return dx


