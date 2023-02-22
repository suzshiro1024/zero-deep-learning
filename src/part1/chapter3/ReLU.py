import numpy as np


def relu(x):
    return np.maximum(x, 0)


if __name__ == "__main__":
    input = np.arange(-2, 3)
    print(f"step([-2,-1,0,1,2])= {relu(input)}")
