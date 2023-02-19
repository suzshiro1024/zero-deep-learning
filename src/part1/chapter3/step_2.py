import numpy as np


def step(x):
    sign = x > 0
    return sign.astype(int)


if __name__ == "__main__":
    input = np.arange(-2, 3)
    print(f"step([-2,-1,0,1,2])= {step(input)}")
