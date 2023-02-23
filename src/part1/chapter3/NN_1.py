import numpy as np


if __name__ == "__main__":
    X = np.array([1, 2])
    W = np.array([[1, 2], [3, 4], [5, 6]])

    Y = np.dot(W, X)
    print(f"Y = {Y}")
