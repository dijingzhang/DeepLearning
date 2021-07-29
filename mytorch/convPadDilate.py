import numpy as np


class Conv2D_Pad_Dilate():
    def __init__(self, in_channel, out_channel,
                 kernel_size, stride, padding=0, dilation=1,
                 weight_init_fn=None, bias_init_fn=None):

        self.in_channel = in_channel
        self.out_channel = out_channel
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        # after doing the dilation
        self.kernel_dilated = (self.kernel_size - 1) * self.dilation + 1

        if weight_init_fn is None:
            self.W = np.random.normal(0, 1.0, (out_channel, in_channel, kernel_size, kernel_size))
        else:
            self.W = weight_init_fn(out_channel, in_channel, kernel_size, kernel_size)

        self.W_dilated = np.zeros((self.out_channel, self.in_channel, self.kernel_dilated, self.kernel_dilated))

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
        # do padding
        x = np.pad(x, pad_width=((0, 0), (0, 0), (self.padding, self.padding), (self.padding, self.padding)))
        self.state = x

        # do dilation
        # first upsample the W
        # computation: k_new = (k-1) * (dilation-1) + k = (k-1) * d + 1
        for i in range(self.kernel_size):
            for j in range(self.kernel_size):
                self.W_dilated[:, :, i * self.dilation, j * self.dilation] = self.W[:, :, i, j]

        bs = x.shape[0]
        out_width = (x.shape[2] - self.kernel_dilated) // self.stride + 1
        out_height = (x.shape[3] - self.kernel_dilated) // self.stride + 1
        out = np.zeros((bs, self.out_channel, out_width, out_height))
        for i in range(out_width):
            for j in range(out_height):
                # segment.shape = [bs, in_channel, k, k]
                segment = x[:, :, i * self.stride: i * self.stride + self.kernel_dilated,
                          j * self.stride: j * self.stride + self.kernel_dilated]
                # W.shape = [out_channel, in_channel, k, k]
                out[:, :, i, j] = np.tensordot(segment, self.W_dilated, axes=([1, 2, 3], [1, 2, 3]))  # [bs, out_channel, 1, 1】
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
        up_width = input_width - self.kernel_dilated + 1
        up_height = input_height - self.kernel_dilated + 1

        dzup = np.zeros((bs, self.out_channel, up_width, up_height))
        if self.stride > 1:
            for i in range(output_width):
                for j in range(output_height):
                    dzup[:, :, i * self.stride, j * self.stride] = delta[:, :, i, j]
        else:
            dzup = delta

        dzpad = np.pad(dzup, pad_width=(
        (0, 0), (0, 0), (self.kernel_dilated - 1, self.kernel_dilated - 1), (self.kernel_dilated - 1, self.kernel_dilated - 1)),
                       mode='constant', constant_values=0)

        # compute the dx
        dx = np.zeros(self.state.shape)
        # flip the W
        W_flip = np.zeros(self.W_dilated.shape)
        for i in range(W_flip.shape[0]):
            for j in range(W_flip.shape[1]):
                # should flip W up-down and left-right
                W_flip[i, j, :, :] = np.flipud(np.fliplr(self.W_dilated[i, j, :, :]))

        for i in range(input_width):
            for j in range(input_height):
                segment = dzpad[:, :, i: i + self.kernel_dilated,
                          j: j + self.kernel_dilated]  # segment.shape=[bs, out_channel, k, k]
                # W_flip = np.flip(self.W, (2, 3))  # W_flip.shape=[out_channel, in_channel, k, k]
                dx[:, :, i, j] = np.tensordot(segment, W_flip, axes=([1, 2, 3], [0, 2, 3]))
        dx = dx[:, 2 * self.padding: -2 * self.padding, 2 * self.padding:-2 * self.padding]

        # compute the dW
        dW_dilated = np.zeros_like(self.W_dilated)
        for i in range(self.kernel_dilated):
            for j in range(self.kernel_dilated):
                segment = self.state[:, :, i: i + up_width,
                          j: j + up_height]  # segment.shape=[bs, in_channel, width, height]
                dW_dilated[:, :, i, j] = np.tensordot(dzup, segment, axes=(
                [0, 2, 3], [0, 2, 3]))  # dzup.shape=[bs, out_channel, width, height]
        for i in range(self.kernel_size):
            for j in range(self.kernel_size):
                self.dW[:, :, i, j] = dW_dilated[:, :, i * self.dilation, j * self.dilation]


        # compute the db
        self.db = np.sum(delta, axis=(0, 2, 3))

        return dx