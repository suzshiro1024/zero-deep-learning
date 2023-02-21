import numpy as np


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


if __name__ == "__main__":
    input = np.arange(-2, 3)
    print(f"step([-2,-1,0,1,2])= {sigmoid(input)}")
