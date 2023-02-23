import numpy as np
import sigmoid


def init_network():
    network = {}
    network["b1"] = np.array([0.1, 0.2, 0.3])
    network["b2"] = np.array([0.1, 0.2])
    network["b3"] = np.array([0.1, 0.2])
    network["w1"] = np.array([[0.1, 0.3, 0.5], [0.2, 0.4, 0.6]])
    network["w2"] = np.array([[0.1, 0.4], [0.2, 0.5], [0.3, 0.6]])
    network["w3"] = np.array([[0.1, 0.3], [0.2, 0.4]])
    return network


def identity_function(x):
    return x


def forward(network, x):
    w1, w2, w3 = network["w1"], network["w2"], network["w3"]
    b1, b2, b3 = network["b1"], network["b2"], network["b3"]

    # phase 1
    a1 = np.dot(x, w1) + b1
    z1 = sigmoid.sigmoid(a1)

    # phase 2
    a2 = np.dot(z1, w2) + b2
    z2 = sigmoid.sigmoid(a2)

    # phase 3
    a3 = np.dot(z2, w3) + b3
    y = identity_function(a3)

    return y


if __name__ == "__main__":
    network = init_network()
    x = np.array([1.0, 0.5])
    y = forward(network, x)
    print(f"x = {x}\ny = {y}")
