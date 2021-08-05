import numpy as np

import sys
sys.path.append('..')
import os

from mytorch import autograd_engine
import mytorch.nn as nn
from mytorch import optim
from mytorch.nn.functional import *

DATA_PATH = "./data"

class MLP(object):

    """
    A simple multilayer perceptron
    """

    def __init__(self, input_size, output_size, hiddens, activations,
                 criterion, lr, autograd_engine, momentum=0.0):

        # Don't change this -->
        self.train_mode = True
        self.nlayers = len(hiddens) + 1
        self.input_size = input_size
        self.output_size = output_size
        self.activations = activations
        self.criterion = criterion
        self.lr = lr
        self.momentum = momentum
        self.autograd_engine = autograd_engine # NOTE: Use this Autograd object for backward
        feature_sizes = [self.input_size] + hiddens + [self.output_size]
        self.linear_layers =[nn.Linear(feature_sizes[i], feature_sizes[i+1], self.autograd) for i in range(len(feature_sizes)-1)]

    def forward(self, x):
        """
        Argument:
            x (np.array): (batch size, input_size)
        Return:
            out (np.array): (batch size, output_size)
        """
        # Complete the forward pass through your entire MLP.
        for l in range(len(self.linear_layers)):
            x = self.linear_layers[l](x)
            x = self.activations[l](x)
        self.output = x
        return x

    def zero_grads(self):
        # Use numpyArray.fill(0.0) to zero out your backpropped derivatives in each
        # of your linear layers.
        for layer in self.linear_layers:
            layer.dW.fill(0.0)
            layer.db.fill(0.0)

        self.autograd_engine.zero_grad()

    def step(self):
        # Apply a step to the weights and biases of the linear layers.
        for i in range(len(self.linear_layers)):
            # Update weights and biases here
            layer = self.linear_layers[i]
            delta_W = self.momentum * layer.momentum_W - self.lr * layer.dW
            layer.W += delta_W
            layer.momentum_W = delta_W

            delta_b = self.momentum * layer.momentum_b - self.lr * layer.db
            layer.b += delta_b
            layer.momentum_b = delta_b

    def backward(self, labels):
        # The magic of autograd: This is 2 lines.
        # Get the loss.
        # Call autograd backward.
        loss = self.criterion(labels, self.output)
        self.autograd_engine.backward(loss)
        # or self.criterion.backward()

    def error(self, labels):
        return (np.argmax(self.output, axis = 1) != np.argmax(labels, axis = 1)).sum()

    def total_loss(self, labels):
        # NOTE: Put the inputs in the correct order for the criterion
        # return self.criterion().sum()
        return self.criterion(labels, self.output).sum()

    def __call__(self, x):
        return self.forward(x)

    def train(self):
        self.train_mode = True

    def eval(self):
        self.train_mode = False

def get_training_stats(mlp, dset, nepochs, batch_size=1):
    # NOTE: Because the batch size is 1 (unless you support
    # broadcasting) the MLP training will be slow.
    train, val, _ = dset
    trainx, trainy = train
    valx, valy = val

    idxs = np.arange(len(trainx))

    training_losses = np.zeros(nepochs)
    training_errors = np.zeros(nepochs)
    validation_losses = np.zeros(nepochs)
    validation_errors = np.zeros(nepochs)

    # Setup ...
    for e in range(nepochs):
        print("Epoch", e)
        # Per epoch setup ...
        mlp.train()
        mlp.zero_grads()
        for b in range(0, len(trainx)):
            # Train ...
            # NOTE: Batchsize is 1 for this bonus unless you support
            # broadcasting/unbroadcasting then you can change this in
            # the mlp_runner.py
            x = np.expand_dims(trainx[b], 0)
            y = np.expand_dims(trainy[b], 0)

            out = mlp(x)
            mlp.backward(y)
            mlp.step()
            mlp.zero_grads()
            training_losses[e] += mlp.total_loss(y)
            training_errors[e] += mlp.error(y)
            if b % 100 == 0:
                print(b, end='\r')
        training_losses[e] /= len(trainx)
        training_errors[e] /= len(trainx)

        mlp.eval()
        for b in range(0, len(valx)):
            # Val ...
            x = np.expand_dims(valx[b], 0)
            y = np.expand_dims(valy[b], 0)
            out = mlp(x)
            validation_losses[e] += mlp.total_loss(y)
            validation_errors[e] += mlp.error(y)
            if b % 100 == 0:
                print(b, end='\r')

        validation_losses[e] /= len(valx)
        validation_errors[e] /= len(valx)

        print("Training Loss:", training_losses[e])
        print("Training Errors:", training_errors[e])
        print("Validation Loss:", validation_losses[e])
        print("Validation Errors:", validation_errors[e])
        # Accumulate data...

        # Cleanup ...

        # Return results ...
    return (training_losses, training_errors, validation_losses, validation_errors)


def load_data():
    train_x = np.load(os.path.join(DATA_PATH, "train_data.npy"))
    train_y = np.load(os.path.join(DATA_PATH, "train_labels.npy"))
    val_x = np.load(os.path.join(DATA_PATH, "val_data.npy"))
    val_y = np.load(os.path.join(DATA_PATH, "val_labels.npy"))

    train_x = train_x / 255
    val_x = val_x / 255

    return train_x, train_y, val_x, val_y


if __name__=='__main__':
    np.random.seed(0)
    ### Testing with random sample for now
    train_x, train_y, val_x, val_y = load_data()
    train = (train_x,train_y)
    val = (val_x,val_y)
    epochs = 5
    autograd = autograd_engine.Autograd()
    hiddens = [256, 128]
    lr = 0.001
    input_size = 784
    output_size = 10
    criterion = nn.loss.SoftmaxCrossEntropy(autograd)
    mlp = MLP(input_size, output_size, hiddens, criterion, lr,autograd)
    train_loss, train_error, valid_loss, valid_error =  get_training_stats(mlp,train,val,epochs)