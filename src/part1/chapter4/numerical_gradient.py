import sys
import os

sys.path.append(os.pardir)

import numpy as np
from chapter4.sample_function2 import func2


def numerical_gradient(f, x):
    h = 1e-4
    grad = np.zeros_like(x)

    for idx in range(x.size):
        tmp_val = x[idx]

        x[idx] = tmp_val + h
        fx1 = f(x)

        x[idx] = tmp_val - h
        fx2 = f(x)

        grad[idx] = (fx1 - fx2) / (2 * h)
        x[idx] = tmp_val

    return grad


if __name__ == "__main__":
    print(f"grad(3,4) = {numerical_gradient(func2, np.array([3.0,4.0]))}")
