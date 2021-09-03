import numpy as np

"""
self.mW = np.zeros(None) #mean derivative for W
self.vW = np.zeros(None) #squared derivative for W
self.mb = np.zeros(None) #mean derivative for b
self.vb = np.zeros(None) #squared derivative for b
"""

class adam():
    def __init__(self, model, beta1=0.9, beta2=0.999, eps=1e-8):
        self.model = model
        self.beta1 = beta1
        self.beta2 = beta2
        self.eps = eps
        self.lr = self.model.lr
        self.t = 0  # Number of Updates

    def step(self):
        '''
        * self.model is an instance of your MLP in hw1/hw1.py, it has access to
          the linear layer's list.
        * Each linear layer is an instance of the Linear class, and thus has
          access to the added class attributes dicussed above as well as the
          original attributes such as the weights and their gradients.
        '''
        self.t += 1
        linear_layers = self.model.linear_layers
        for i in range(self.model.nlayers):
            linear = linear_layers[i]
            gt_w = linear.dW
            gt_b = linear.db

            linear.mW = self.beta1 * linear.mW + (1 - self.beta1) * gt_w
            linear.mb = self.beta1 * linear.mb + (1 - self.beta1) * gt_b
            linear.vW = self.beta2 * linear.vW + (1 - self.beta2) * np.power(gt_w, 2)
            linear.vb = self.beta2 * linear.vb + (1 - self.beta2) * np.power(gt_b, 2)

            mt_w_head = linear.mW / (1 - np.power(self.beta1, self.t))
            mt_b_head = linear.mb / (1 - np.power(self.beta1, self.t))
            vt_w_head = linear.vW / (1 - np.power(self.beta2, self.t))
            vt_b_head = linear.vb / (1 - np.power(self.beta2, self.t))

            linear.W -= self.lr * mt_w_head / (np.sqrt(vt_w_head + self.eps))
            linear.b -= self.lr * mt_b_head / (np.sqrt(vt_b_head + self.eps))
