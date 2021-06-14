import numpy as np
W = np.random.random((4, 5, 3))

W_flip = np.zeros_like(W)
for i in range(W.shape[0]):  # W=[out_channel, in_channel, kernel_size]
    W_flip[i, :, :] = np.fliplr(W[i, :, :])
    print(W_flip[i, :, :])
    print(W[i, :, :])
    break