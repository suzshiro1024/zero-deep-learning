import numpy as np


class Affine:
    def __init__(self, W, b):
        self.W = W
        self.b = b

        self.x = None
        self.original_x_shape = None
        self.d_W = None
        self.d_b = None

    def forward(self, x):
        # テンソル対応
        self.original_x_shape = x.shape
        x = x.reshape(x.shape[0], -1)
        self.x = x

        out = np.dot(self.x, self.W) + self.b

        return out

    def backward(self, d_output):
        d_x = np.dot(d_output, self.W.T)
        self.d_W = np.dot(self.x.T, d_output)
        self.d_b = np.sum(d_output, axis=0)

        d_x = d_x.reshape(*self.original_x_shape)
        return d_x
