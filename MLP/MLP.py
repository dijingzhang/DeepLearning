"""
Notes:

The __call__ method is a special reserved method in
python that defines the behaviour of an object when it is
used as a function. For example, take the Linear activation
function whose implementation has been provided.

# >>> activation = Identity()
# >>> activation(3)
# 3
# >>> activation.forward(3)
# 3
"""

import numpy as np
import sys

sys.path.append('mytorch')
from loss import *
from activation import *
from batchnorm import *
from linear import *


class MLP(object):

    """
    A simple multilayer perceptron
    """

    def __init__(self, input_size, output_size, hiddens, activations, weight_init_fn,
                 bias_init_fn, criterion, lr, momentum=0.0, num_bn_layers=0):

        # Don't change this -->
        self.train_mode = True
        self.num_bn_layers = num_bn_layers
        self.bn = num_bn_layers > 0
        self.nlayers = len(hiddens) + 1
        self.input_size = input_size
        self.output_size = output_size
        self.activations = activations
        self.criterion = criterion
        self.lr = lr
        self.momentum = momentum
        # <---------------------

        # Don't change the name of the following class attributes,
        # the autograder will check against these attributes. But you will need to change
        # the values in order to initialize them correctly

        # Initialize and add all your linear layers into the list 'self.linear_layers'
        # (HINT: self.foo = [ bar(???) for ?? in ? ])
        # (HINT: Can you use zip here?)
        li_layers = []
        li_layers.append(self.input_size)
        li_layers.extend(hiddens)
        li_layers.append(output_size)
        self.linear_layers = [Linear(li_layers[i], li_layers[i+1], weight_init_fn, bias_init_fn) for i in range(self.nlayers)]


        # If batch norm, add batch norm layers into the list 'self.bn_layers'
        if self.bn:
            self.bn_layers = [BatchNorm(li_layers[i+1]) for i in range(self.num_bn_layers)]

    def forward(self, x):
        """
        Argument:
            x (np.array): (batch size, input_size)
        Return:
            out (np.array): (batch size, output_size)
        """
        # Complete the forward pass through your entire MLP.
        input = x
        if not self.bn:
            for i in range(self.nlayers):
                linear = self.linear_layers[i]
                y = linear.forward(input)
                activation = self.activations[i]
                output = activation.forward(y)
                input = output
        else:
            for i in range(self.nlayers):
                linear = self.linear_layers[i]
                y = linear.forward(input)
                activation = self.activations[i]
                if i <= self.num_bn_layers - 1:
                    bn_layer = self.bn_layers[i]
                    if self.train_mode:
                        y = bn_layer.forward(y)
                    else:
                        y = bn_layer.forward(y, eval=True)
                output = activation.forward(y)
                input = output

        self.output = input
        return self.output

    def zero_grads(self):
        # Use numpyArray.fill(0.0) to zero out your backpropped derivatives in each
        # of your linear and batchnorm layers.
        for i in range(self.nlayers):
            linear = self.linear_layers[i]
            linear.dW.fill(0.0)
            linear.db.fill(0.0)
        for j in range(self.num_bn_layers):
            bn_layer = self.bn_layers[j]
            bn_layer.dgamma.fill(0.0)
            bn_layer.dbeta.fill(0.0)

    def step(self):
        # Apply a step to the weights and biases of the linear layers.
        # Apply a step to the weights of the batchnorm layers.
        # (You will add momentum later in the assignment to the linear layers only
        # , not the batchnorm layers)

        for i in range(self.nlayers):
            linear = self.linear_layers[i]
            if self.momentum == 0.0:
                linear.W -= linear.dW * self.lr
                linear.b -= linear.db * self.lr
            else:
                linear.momentum_W = linear.momentum_W * self.momentum - self.lr * linear.dW
                linear.momentum_b = linear.momentum_b * self.momentum - self.lr * linear.db
                linear.W += linear.momentum_W
                linear.b += linear.momentum_b

        # Do the same for batchnorm layers
        for j in range(self.num_bn_layers):
            bn_layer = self.bn_layers[j]
            bn_layer.gamma -= self.lr * bn_layer.dgamma
            bn_layer.beta -= self.lr * bn_layer.dbeta

    def backward(self, labels):
        # Backpropagate through the activation functions, batch norm and
        # linear layers.
        # Be aware of which return derivatives and which are pure backward passes
        # i.e. take in a loss w.r.t it's output.
        self.criterion(self.output, labels)
        deriv = self.criterion.derivative()
        if self.bn:
            for i in range(self.nlayers-1, -1, -1):
                linear = self.linear_layers[i]
                activation = self.activations[i]
                delta = deriv * activation.derivative()
                if i <= self.num_bn_layers - 1:
                    bn_layer = self.bn_layers[i]
                    dx = bn_layer.backward(delta)
                    dx = linear.backward(dx)
                else:
                    dx = linear.backward(delta)
                deriv = dx

        else:
            for i in range(self.nlayers-1, -1, -1):
                linear = self.linear_layers[i]
                activation = self.activations[i]
                delta = deriv * activation.derivative()
                dx = linear.backward(delta)
                deriv = dx

    def error(self, labels):
        return (np.argmax(self.output, axis = 1) != np.argmax(labels, axis = 1)).sum()

    def total_loss(self, labels):
        return self.criterion(self.output, labels).sum()

    def __call__(self, x):
        return self.forward(x)

    def train(self):
        self.train_mode = True

    def eval(self):
        self.train_mode = False

