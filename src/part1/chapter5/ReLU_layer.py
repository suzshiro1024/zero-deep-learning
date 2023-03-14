import numpy as np


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
