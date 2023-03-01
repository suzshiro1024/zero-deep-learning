import numpy as np


def cross_entropy_error(y, t):
    delta = 1e-7
    return -1 * np.sum(t * np.log(y + delta))


if __name__ == "__main__":
    t = np.array([0, 0, 1, 0, 0, 0, 0, 0, 0, 0])
    y1 = np.array([0.1, 0.05, 0.6, 0.0, 0.05, 0.1, 0.0, 0.1, 0.0, 0.0])
    y2 = np.array([0.1, 0.05, 0.1, 0.0, 0.05, 0.1, 0.0, 0.6, 0.0, 0.0])

    e1 = cross_entropy_error(y1, t)
    e2 = cross_entropy_error(y2, t)

    print(f"e1 = {e1}\ne2 = {e2}")
