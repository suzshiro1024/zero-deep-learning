import numpy as np


def sum_squared_error(y, t):
    return 0.5 * np.sum((y - t) ** 2)


if __name__ == "__main__":
    t = np.array([0, 0, 1, 0, 0, 0, 0, 0, 0, 0])
    y1 = np.array([0.1, 0.05, 0.6, 0.0, 0.05, 0.1, 0.0, 0.1, 0.0, 0.0])
    y2 = np.array([0.1, 0.05, 0.1, 0.0, 0.05, 0.1, 0.0, 0.6, 0.0, 0.0])

    e1 = sum_squared_error(y1, t)
    e2 = sum_squared_error(y2, t)

    print(f"e1 = {e1}\ne2 = {e2}")
