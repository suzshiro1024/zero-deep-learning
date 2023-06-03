import sys
import os

sys.path.append(os.pardir)

import numpy as np
from common.layers import *
from common.convolution import Convolution
from common.pooling import Pooling
from collections import OrderedDict


class SimpleConvNet:
    def __init__(
        self,
        input_dim=(1, 28, 28),
        conv_param={"filter_num": 30, "filter_size": 5, "padding": 0, "stride": 1},
        hidden_size=100,
        output_size=10,
        weight_init_std=0.01,
    ):
        filter_num = conv_param["filter_num"]
        filter_size = conv_param["filter_size"]
        filter_padding = conv_param["padding"]
        filter_stride = conv_param["stride"]

        input_size = input_dim[1]
        conv_output_size = (
            input_size - filter_size + 2 * filter_padding
        ) / filter_stride + 1
        pool_output_size = int(
            filter_num * (conv_output_size / 2) * (conv_output_size / 2)
        )

        self.params = {}
        self.params["W1"] = weight_init_std * np.random.randn(
            filter_num, input_dim[0], filter_size, filter_size
        )
        self.params["b1"] = np.zeros(filter_num)
        self.params["W2"] = weight_init_std * np.random.randn(
            pool_output_size, hidden_size
        )
        self.params["b2"] = np.zeros(hidden_size)
        self.params["W3"] = weight_init_std * np.random.randn(hidden_size, output_size)
        self.params["b3"] = np.zeros(output_size)

        self.layers = OrderedDict()

        self.layers["Conv1"] = Convolution(
            self.params["W1"],
            self.params["b1"],
            conv_param["stride"],
            conv_param["padding"],
        )
        self.layers["ReLU1"] = ReLU()
        self.layers["Pool1"] = Pooling(pool_h=2, pool_w=2, stride=2)
        self.layers["Affine1"] = Affine(self.params["W2"], self.params["b2"])
        self.layers["ReLU2"] = ReLU()
        self.layers["Affine2"] = Affine(self.params["W3"], self.params["b3"])
        self.lastLayer = softmaxWithLoss()

    def predict(self, x):
        for layer in self.layers.values():
            x = layer.forward(x)
        return x

    def loss(self, x, t):
        y = self.predict(x)
        return self.lastLayer.forward(y, t)

    def gradient(self, x, t):
        self.loss(x, t)

        d_output = 1
        d_output = self.lastLayer.backward(d_output)

        layers = list(self.layers.values())
        layers.reverse()
        for layer in layers:
            d_output = layer.backward(d_output)

        grads = {}
        grads["W1"] = self.layers["Conv1"].d_W
        grads["b1"] = self.layers["Conv1"].d_b
        grads["W2"] = self.layers["Affine1"].d_W
        grads["b2"] = self.layers["Affine1"].d_b
        grads["W3"] = self.layers["Affine2"].d_W
        grads["b3"] = self.layers["Affine2"].d_b

        return grads
