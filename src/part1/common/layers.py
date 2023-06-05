import sys
import os

sys.path.append(os.pardir)

from common.cross_entropy_error_batch import cross_entropy_error
from common.soft_max import soft_max


import numpy as np


class addLayer:
    def __init__(self):
        pass

    def forward(self, x, y):
        return x + y

    def backward(self, d_output):
        d_x = d_output * 1
        d_y = d_output * 1
        return d_x, d_y


class mulLayer:
    def __init__(self):
        self.x = None
        self.y = None

    def forward(self, x, y):
        self.x = x
        self.y = y
        return x * y

    def backward(self, d_output):
        d_x = d_output * self.y
        d_y = d_output * self.x
        return d_x, d_y


class ReLU:
    def __init__(self):
        self.mask = None

    def forward(self, x):
        self.mask = x <= 0
        out = x.copy()
        out[self.mask] = 0
        return out

    def backward(self, d_output):
        d_output[self.mask] = 0
        d_x = d_output
        return d_x


class Sigmoid:
    def __init__(self):
        self.out = None

    def forward(self, x):
        out = 1 / (1 + np.exp(-x))

        self.out = out
        return out

    def backward(self, d_output):
        d_x = d_output * self.out * (1 - self.out)
        return d_x


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


class softmaxWithLoss:
    def __init__(self):
        self.loss = None
        self.y = None
        self.t = None

    def forward(self, x, t):
        self.t = t
        self.y = soft_max(x)
        self.loss = cross_entropy_error(self.y, self.t)

        return self.loss

    def backward(self, d_output=1):
        batch_size = self.t.shape[0]
        d_x = (self.y - self.t) / batch_size

        return d_x
