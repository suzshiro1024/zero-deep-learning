import sys
import os

sys.path.append(os.pardir)

import numpy as np
from cross_entropy_error_batch import cross_entropy_error
from chapter3.sigmoid import sigmoid
from chapter3.soft_max import soft_max


def numerical_gradient(f, x):
    h = 1e-4  # 0.0001
    grad = np.zeros_like(x)

    it = np.nditer(x, flags=["multi_index"], op_flags=["readwrite"])
    while not it.finished:
        idx = it.multi_index
        tmp_val = x[idx]
        x[idx] = tmp_val + h
        fxh1 = f(x)  # f(x+h)

        x[idx] = tmp_val - h
        fxh2 = f(x)  # f(x-h)
        grad[idx] = (fxh1 - fxh2) / (2 * h)

        x[idx] = tmp_val  # 値を元に戻す
        it.iternext()

    return grad


class TwoLayerNet:
    def __init__(self, input_size, hidden_size, output_size, weight_init_std=0.01):
        self.params = {}
        self.params["W1"] = weight_init_std * np.random.randn(input_size, hidden_size)
        self.params["b1"] = np.zeros(hidden_size)
        self.params["W2"] = weight_init_std * np.random.randn(hidden_size, output_size)
        self.params["b2"] = np.zeros(output_size)

    def predict(self, x):
        W1, W2 = self.params["W1"], self.params["W2"]
        b1, b2 = self.params["b1"], self.params["b2"]

        a1 = np.dot(x, W1) + b1
        z1 = sigmoid(a1)

        a2 = np.dot(z1, W2) + b2
        y = soft_max(a2)

        return y

    def loss(self, x, t):
        y = self.predict(x)

        return cross_entropy_error(y, t)

    def accuracy(self, x, t):
        y = self.predict(x)
        y = np.argmax(y, axis=1)
        t = np.argmax(t, axis=1)

        accuracy = np.sum(y == t) / float(x.shape[0])
        return accuracy

    def numerical_gradient(self, x, t):
        loss_W = lambda W: self.loss(x, t)

        grads = {}
        grads["W1"] = numerical_gradient(loss_W, self.params["W1"])
        grads["b1"] = numerical_gradient(loss_W, self.params["b1"])
        grads["W2"] = numerical_gradient(loss_W, self.params["W2"])
        grads["b2"] = numerical_gradient(loss_W, self.params["b2"])

        return grads


if __name__ == "__main__":
    net1 = TwoLayerNet(input_size=784, hidden_size=100, output_size=10)
    print(f"param[W1] = {net1.params['W1'].shape}")
    print(f"param[b1] = {net1.params['b1'].shape}")
    print(f"param[W2] = {net1.params['W2'].shape}")
    print(f"param[b2] = {net1.params['b2'].shape}")

    x = np.random.rand(100, 784)
    t = np.random.rand(100, 10)
    y = net1.predict(x)

    grads = net1.numerical_gradient(x, t)

    print(f"grads[W1] = {grads['W1'].shape}")
    print(f"grads[b1] = {grads['b1'].shape}")
    print(f"grads[W2] = {grads['W2'].shape}")
    print(f"grads[b2] = {grads['b2'].shape}")
