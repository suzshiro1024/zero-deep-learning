import sys
import os

sys.path.append(os.pardir)

from chapter4.cross_entropy_error_batch import cross_entropy_error
from chapter3.soft_max import soft_max


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
