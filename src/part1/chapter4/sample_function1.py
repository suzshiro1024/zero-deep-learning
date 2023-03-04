import numpy as np
import matplotlib.pyplot as plt


def func1(x):
    return 0.01 * x ** 2 + 0.1 * x


if __name__ == "__main__":
    x = np.arange(0.0, 20.0, 0.1)
    y = func1(x)

    plt.xlabel("x")
    plt.ylabel("f(x)")
    plt.plot(x, y)
    plt.savefig("../../../figure/sample_function1.png")
