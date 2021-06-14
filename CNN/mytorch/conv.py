# Do not import any additional 3rd party external libraries as they will not
# be available to AutoLab and are not needed (or allowed)

import numpy as np


class Conv1D():
    def __init__(self, in_channel, out_channel, kernel_size, stride,
                 weight_init_fn=None, bias_init_fn=None):
        # Do not modify this method
        self.in_channel = in_channel
        self.out_channel = out_channel
        self.kernel_size = kernel_size
        self.stride = stride

        if weight_init_fn is None:
            # random initialization
            self.W = np.random.normal(0, 1.0, (out_channel, in_channel, kernel_size))
        else:
            self.W = weight_init_fn(out_channel, in_channel, kernel_size)
        
        if bias_init_fn is None:
            # zero initialization
            self.b = np.zeros(out_channel)
        else:
            self.b = bias_init_fn(out_channel)

        self.dW = np.zeros(self.W.shape)
        self.db = np.zeros(self.b.shape)

        self.state = None

    def __call__(self, x):
        return self.forward(x)

    def forward(self, x):
        """
        Argument:
            x (np.array): (batch_size, in_channel, input_size)
        Return:
            out (np.array): (batch_size, out_channel, output_size)
        """
        self.state = x

        bs, input_size = x.shape[0], x.shape[2]
        output_size = (input_size - self.kernel_size) // self.stride + 1
        out = np.zeros((bs, self.out_channel, output_size))

        for i in range(output_size):
            segment = x[:, :, i * self.stride: i * self.stride + self.kernel_size]  # [bs, in_channel, kernel]
            out[:, :, i] = np.tensordot(segment, self.W, axes=([1, 2], [1, 2])) + self.b   # [bs, out_channel, 1]. W=[out_channel, in_channel, kernel]
        return out

    def backward(self, delta):
        """
        Argument:
            delta (np.array): (batch_size, out_channel, output_size)
        Return:
            dx (np.array): (batch_size, in_channel, input_size)
        """
        bs, output_size = delta.shape[0], delta.shape[2]
        input_size = self.state.shape[2]

        # upsample dz
        # (dz is the matrix taking stride = 1. If stride > 1, then we should zero the intervals)
        Wup = input_size - self.kernel_size + 1

        if self.stride > 1:
            dzup = np.zeros((bs, self.out_channel, Wup))
            for i in range(output_size):
                dzup[:, :, i * self.stride] = delta[:, :, i]  # [bs, out_channel, Wup]
        else:
            dzup = delta
        dzpad = np.pad(dzup, pad_width=((0, 0), (0, 0), (self.kernel_size - 1, self.kernel_size - 1)),
                       mode='constant', constant_values=0)

        # compute the dx
        dx = np.zeros((bs, self.in_channel, input_size))
        # flip the W
        W_flip = np.zeros(self.W.shape)
        for i in range(self.W.shape[0]):  # W=[out_channel, in_channel, kernel_size]
            W_flip[i, :, :] = np.fliplr(self.W[i, :, :])  # W_flip=[out_channel, in_channel, kernel_size], which flips left and right
        for i in range(input_size):
            segment = dzpad[:, :, i: i + self.kernel_size]  # [bs, out_channel, kernel_size]
            dx[:, :, i] = np.tensordot(segment,  W_flip, axes=([1, 2], [0, 2]))  # dx=[bs, in_channel, 1]

        # compute the dW
        for j in range(self.kernel_size):
            segment = self.state[:, :, j: j + Wup]   # [bs, in_channel, Width]
            self.dW[:, :, j] = np.tensordot(dzup, segment, axes=([0, 2], [0, 2]))  # [out_channel, in_channel]

        # compute the db
        self.db = np.sum(delta, axis=(0, 2))  # [out_channel]

        return dx


class Conv2D():
    def __init__(self, in_channel, out_channel,
                 kernel_size, stride,
                 weight_init_fn=None, bias_init_fn=None):

        self.in_channel = in_channel
        self.out_channel = out_channel
        self.kernel_size = kernel_size
        self.stride = stride

        if weight_init_fn is None:
            self.W = np.random.normal(0, 1.0, (out_channel, in_channel, kernel_size, kernel_size))
        else:
            self.W = weight_init_fn(out_channel, in_channel, kernel_size, kernel_size)
        
        if bias_init_fn is None:
            self.b = np.zeros(out_channel)
        else:
            self.b = bias_init_fn(out_channel)

        self.dW = np.zeros(self.W.shape)
        self.db = np.zeros(self.b.shape)

        self.state = None


    def __call__(self, x):
        return self.forward(x)

    def forward(self, x):
        """
        Argument:
            x (np.array): (batch_size, in_channel, input_width, input_height)
        Return:
            out (np.array): (batch_size, out_channel, output_width, output_height)
        """
        self.state = x

        bs = x.shape[0]
        out_width = (x.shape[2] - self.kernel_size) // self.stride + 1
        out_height = (x.shape[3] - self.kernel_size) // self.stride + 1
        out = np.zeros((bs, self.out_channel, out_width, out_height))
        for i in range(out_width):
            for j in range(out_height):
                # segment.shape = [bs, in_channel, k, k]
                segment = x[:, :, i * self.stride: i * self.stride + self.kernel_size, j * self.stride: j * self.stride + self.kernel_size]
                # W.shape = [out_channel, in_channel, k, k]
                out[:, :, i, j] = np.tensordot(segment, self.W, axes=([1, 2, 3], [1, 2, 3]))  # [bs, out_channel, 1, 1】
        return out

    def backward(self, delta):
        """
        Argument:
            delta (np.array): (batch_size, out_channel, output_width, output_height)
        Return:
            dx (np.array): (batch_size, in_channel, input_width, input_height)
        """
        bs = delta.shape[0]
        output_width, output_height = delta.shape[2], delta.shape[3]
        input_width, input_height = self.state.shape[2], self.state.shape[3]

        # upsample dz
        up_width = input_width - self.kernel_size + 1
        up_height = input_height - self.kernel_size + 1

        dzup = np.zeros((bs, self.out_channel, up_width, up_height))
        if self.stride > 1:
            for i in range(output_width):
                for j in range(output_height):
                    dzup[:, :, i * self.stride, j * self.stride] = delta[:, :, i, j]
        else:
            dzup = delta

        dzpad = np.pad(dzup, pad_width=((0, 0), (0, 0), (self.kernel_size - 1, self.kernel_size - 1), (self.kernel_size - 1, self.kernel_size - 1)),
                       mode='constant', constant_values=0)

        # compute the dx
        dx = np.zeros(self.state.shape)
        # flip the W
        W_flip = np.zeros(self.W.shape)
        for i in range(W_flip.shape[0]):
            for j in range(W_flip.shape[1]):
                # should flip W up-down and left-right
                W_flip[i, j, :, :] = np.flipud(np.fliplr(self.W[i, j, :, :]))

        for i in range(input_width):
            for j in range(input_height):
                segment = dzpad[:, :, i: i + self.kernel_size, j: j + self.kernel_size]  # segment.shape=[bs, out_channel, k, k]
                # W_flip = np.flip(self.W, (2, 3))  # W_flip.shape=[out_channel, in_channel, k, k]
                dx[:, :, i, j] = np.tensordot(segment, W_flip, axes=([1, 2, 3], [0, 2, 3]))

        # compute the dW
        for i in range(self.kernel_size):
            for j in range(self.kernel_size):
                segment = self.state[:, :, i: i + up_width, j: j + up_height]  #segment.shape=[bs, in_channel, width, height]
                self.dW[:, :, i, j] = np.tensordot(dzup, segment, axes=([0, 2, 3],[0, 2, 3]))  #dzup.shape=[bs, out_channel, width, height]

        # compute the db
        self.db = np.sum(delta, axis=(0, 2, 3))

        return dx
        

class Flatten():
    def __call__(self, x):
        return self.forward(x)

    def forward(self, x):
        """
        Argument:
            x (np.array): (batch_size, in_channel, in_width)
        Return:
            out (np.array): (batch_size, in_channel * in width)
        """

        self.b, self.c, self.w = x.shape
        return x.flatten().reshape((self.b, self.c * self.w))

    def backward(self, delta):
        """
        Argument:
            delta (np.array): (batch size, in channel * in width)
        Return:
            dx (np.array): (batch size, in channel, in width)
        """
        return delta.reshape((self.b, self.c, self.w))
