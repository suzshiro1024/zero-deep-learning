import numpy as np


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
