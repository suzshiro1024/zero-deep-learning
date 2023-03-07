import sys
import os

sys.path.append(os.pardir)

import numpy as np
from chapter3.soft_max import soft_max
from cross_entropy_error_batch import cross_entropy_error


class simpleNet:
    def __init__(self):
        self.W = np.random.randn(2, 3)

    def predict(self, x):
        return np.dot(x, self.W)

    def loss(self, x, t):
        z = self.predict(x)
        y = soft_max(z)
        loss = cross_entropy_error(y, t)

        return loss


def numerical_gradient(f, x):
    h = 1e-4
    grad = np.zeros_like(x)

    it = np.nditer(x, flags=["multi_index"], op_flags=["readwrite"])
    while not it.finished:
        idx = it.multi_index
        tmp_val = x[idx]
        x[idx] = tmp_val + h
        fxh1 = f(x)

        x[idx] = tmp_val - h
        fxh2 = f(x)
        grad[idx] = (fxh1 - fxh2) / (2 * h)

        x[idx] = tmp_val
        it.iternext()

    return grad


if __name__ == "__main__":
    x = np.array([0.6, 0.9])
    t = np.array([0, 0, 1])

    net = simpleNet()

    f = lambda w: net.loss(x, t)
    dW = numerical_gradient(f, net.W)

    print(f"dW = {dW}")
