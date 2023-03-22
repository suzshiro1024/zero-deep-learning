import sys
import os

sys.path.append(os.pardir)

import numpy as np
from numerical_gradient import numerical_gradient
from collections import OrderedDict
from chapter5.affine_layer import Affine
from chapter5.ReLU_layer import ReLU
from chapter5.softmax_with_loss import softmaxWithLoss


class TwoLayerNet:
    def __init__(self, input_size, hidden_size, output_size, weight_init_std=0.01):
        self.params = {}
        self.params["W1"] = weight_init_std * np.random.randn(input_size, hidden_size)
        self.params["b1"] = np.zeros(hidden_size)
        self.params["W2"] = weight_init_std * np.random.randn(hidden_size, output_size)
        self.params["b2"] = np.zeros(output_size)

        self.layers = OrderedDict()
        self.layers["Affine1"] = Affine(self.params["W1"], self.params["b1"])
        self.layers["RelU1"] = ReLU()
        self.layers["Affine2"] = Affine(self.params["W2"], self.params["b2"])

        self.lastLayer = softmaxWithLoss()

    def predict(self, x):
        for layer in self.layers.values():
            x = layer.forward(x)

        return x

    def loss(self, x, t):
        y = self.predict(x)
        return self.lastLayer.forward(y, t)

    def accuracy(self, x, t):
        y = self.predict(x)
        y = np.argmax(y, axis=1)
        if t.ndim != 1:
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

    def gradient(self, x, t):
        self.loss(x, t)

        d_output = 1
        d_output = self.lastLayer.backward(d_output)

        layers = list(self.layers.values())
        layers.reverse()
        for layer in layers:
            d_output = layer.backward(d_output)

        grads = {}
        grads["W1"] = self.layers["Affine1"].d_W
        grads["b1"] = self.layers["Affine1"].d_b
        grads["W2"] = self.layers["Affine2"].d_W
        grads["b2"] = self.layers["Affine2"].d_b

        return grads
