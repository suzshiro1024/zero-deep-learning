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
