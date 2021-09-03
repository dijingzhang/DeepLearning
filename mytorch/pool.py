import numpy as np

class MaxPoolLayer():

    def __init__(self, kernel, stride):
        self.kernel = kernel
        self.stride = stride
        self.state = None
        self.arg_max = None
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

        bs, in_channel, input_width, input_height = x.shape[:]
        out_channel = in_channel
        output_width = (input_width-self.kernel) // self.stride + 1
        output_height = (input_height - self.kernel) // self.stride + 1
        out = np.zeros((bs, out_channel, output_width, output_height))

        arg_max = np.zeros(out.shape, dtype=np.int32)

        for l in range(bs):
            for k in range(in_channel):
                m = 0
                for i in range(0, input_width-self.kernel, self.stride):
                    n = 0
                    for j in range(0, input_height - self.kernel, self.stride):
                        segment = x[l, k, i: i+self.kernel, j: j+self.kernel]
                        out[l, k, m, n] = np.max(segment)
                        arg_max[l, k, m, n] = np.argmax(segment)
                        n += 1
                    m += 1

        self.arg_max = arg_max
        return out

    def backward(self, delta):
        """
        Argument:
            delta (np.array): (batch_size, out_channel, output_width, output_height)
        Return:
            dx (np.array): (batch_size, in_channel, input_width, input_height)
        """

        dx = np.zeros(self.state.shape)
        bs, out_channel, out_width, out_height = delta.shape[:]
        kernel_size = (self.kernel, self.kernel)
        for l in range(bs):
            for k in range(out_channel):
                for i in range(out_width):
                    for j in range(out_height):
                        start_i = i * self.stride
                        start_j = j * self.stride
                        end_i = start_i + self.kernel
                        end_j = start_j + self.kernel
                        index = np.unravel_index(self.arg_max[l, k, i, j], kernel_size)
                        dx[l, k, start_i:end_i, start_j:end_j][index] = delta[l, k, i, j]
        return dx


class MeanPoolLayer():

    def __init__(self, kernel, stride):
        self.kernel = kernel
        self.stride = stride
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

        bs, in_channel, input_width, input_height = x.shape[:]
        out_channel = in_channel
        output_width = (input_width - self.kernel) // self.stride + 1
        output_height = (input_height - self.kernel) // self.stride + 1
        out = np.zeros((bs, out_channel, output_width, output_height))

        m = 0
        for i in range(0, input_width - self.kernel, self.stride):
            n = 0
            for j in range(0, input_height - self.kernel, self.stride):
                segment = x[:, :, i: i + self.kernel, j: j + self.kernel]
                out[:, :, m, n] = np.mean(segment, axis=(2, 3))
                n += 1
            m += 1
        return out

    def backward(self, delta):
        """
        Argument:
            delta (np.array): (batch_size, out_channel, output_width, output_height)
        Return:
            dx (np.array): (batch_size, in_channel, input_width, input_height)
        """
        dx = np.zeros(self.state.shape)
        bs, out_channel, out_width, out_height = delta.shape[:]
        for i in range(out_width):
            n = i * self.stride
            for j in range(out_height):
                m = j * self.stride
                for k in range(self.kernel):
                    for l in range(self.kernel):
                        dx[:, :, n+k, m+l] += (1 / (self.kernel ** 2)) * delta[:, :, i, j]
        return dx