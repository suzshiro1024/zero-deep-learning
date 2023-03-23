import sys
import os
import time

sys.path.append(os.pardir)

import numpy as np
from dataset.mnist import load_mnist
from two_layer_net import TwoLayerNet

if __name__ == "__main__":
    (x_train, t_train), (x_test, t_test) = load_mnist(
        normalize=True, one_hot_label=True
    )

    network = TwoLayerNet(input_size=784, hidden_size=50, output_size=10)

    x_batch = x_train[:3]
    t_batch = t_train[:3]

    print("Numerical Gradient Process ...")
    ng_start = time.perf_counter()
    grad_numerical = network.numerical_gradient(x_batch, t_batch)
    print("Done")
    ng_end = time.perf_counter()

    print("Backprop Gradient Process ...")
    bg_start = time.perf_counter()
    grad_backprop = network.gradient(x_batch, t_batch)
    print("Done")
    bg_end = time.perf_counter()

    print("----------Diff----------")
    for key in grad_numerical.keys():
        diff = np.average(np.abs(grad_backprop[key] - grad_numerical[key]))
        print(f"{key}: {diff}")

    print("----------Processing Time----------")
    print(f"Numerical Gradient: {ng_end-ng_start} sec")
    print(f"Backprop Gradient : {bg_end-bg_start} sec")
