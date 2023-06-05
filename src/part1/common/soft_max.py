import numpy as np


def soft_max_naive(x):
    exp = np.exp(x)

    y = exp / np.sum(exp)
    return y


def soft_max(x):
    x = x - np.max(x, axis=-1, keepdims=True)
    return np.exp(x) / np.sum(np.exp(x), axis=-1, keepdims=True)


if __name__ == "__main__":
    a = np.array([0.3, 2.9, 4.0])
    y1 = soft_max_naive(a)
    y2 = soft_max(a)

    print(f"a={a}\ny1={y1}\ny2={y2}")
