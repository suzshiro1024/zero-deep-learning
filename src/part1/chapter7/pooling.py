import sys
import os

sys.path.append(os.pardir)

from common.im2col import im2col
from common.col2im import col2im

import numpy as np


class Pooling:
    def __init__(self, pool_h, pool_w, stride=2, padding=0):
        self.pool_h = pool_h
        self.pool_w = pool_w
        self.stride = stride
        self.padding = padding

        self.x = None
        self.arg_max = None

    def forward(self, x):
        N, C, H, W = x.shape
        out_h = int(1 + (H - self.pool_h) / self.stride)
        out_w = int(1 + (W - self.pool_w) / self.stride)

        col = im2col(x, self.pool_h, self.pool_w, self.stride, self.padding)
        col = col.reshape(-1, self.pool_h * self.pool_w)

        arg_max = np.argmax(col, axis=1)
        out = np.max(col, axis=1)
        out = out.reshape(N, out_h, out_w, C).transpose(0, 3, 1, 2)

        self.x = x
        self.arg_max = arg_max

        return out

    def backward(self, d_output):
        d_output = d_output.transpose(0, 2, 3, 1)

        pool_size = self.pool_h * self.pool_w
        d_max = np.zeros((d_output.size, pool_size))
        d_max[np.arange(self.arg_max.size), self.arg_max.flatten()] = d_output.flatten()
        d_max = d_max.reshape(d_output.shape + (pool_size,))

        d_col = d_max.reshape(d_max.shape[0] * d_max.shape[1] * d_max.shape[2], -1)
        d_x = col2im(
            d_col, self.x.shape, self.pool_h, self.pool_w, self.stride, self.padding
        )

        return d_x
