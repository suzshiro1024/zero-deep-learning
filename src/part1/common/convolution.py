import sys
import os

sys.path.append(os.pardir)

from common.im2col import im2col
from common.col2im import col2im

import numpy as np


class Convolution:
    def __init__(self, W, b, stride=1, padding=0):
        self.W = W
        self.b = b
        self.stride = stride
        self.padding = padding

        self.x = None
        self.col = None
        self.col_W = None

        self.d_W = None
        self.d_b = None

    def forward(self, x):
        # N:batch_size, C:channel, H:height, W:weight
        FN, C, FH, FW = self.W.shape
        N, C, H, W = x.shape

        out_h = int(1 + (H + 2 * self.padding - FH) / self.stride)
        out_w = int(1 + (W + 2 * self.padding - FW) / self.stride)

        col = im2col(x, FH, FW, self.stride, self.padding)
        col_W = self.W.reshape(FN, -1).T

        out = np.dot(col, col_W) + self.b

        out = out.reshape(N, out_h, out_w, -1).transpose(0, 3, 1, 2)

        return out

    def backward(self, d_output):
        FN, C, FH, FW = self.W.shape
        d_output = d_output.transpose(0, 2, 3, 1).reshape(-1, FN)

        self.d_b = np.sum(d_output, axis=0)
        self.d_W = np.dot(self.col.T, d_output)
        self.d_W = self.d_W.transpose(1, 0).reshape(FN, C, FH, FW)

        d_col = np.dot(d_output, self.col_W.T)
        d_x = col2im(d_col, self.x.shape, FH, FW, self.stride, self.padding)

        return d_x
