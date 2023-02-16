import numpy as np


def or_gate(x1, x2):
    # y = 0 (bias + w.Tx <= 0)
    # y = 1 (bias + w.Tx > 0)
    x = np.array([x1, x2])
    w = np.array([0.5, 0.5])
    bias = -0.2
    tmp = np.sum(w * x) + bias

    if tmp <= 0:
        return 0
    elif tmp > 0:
        return 1


if __name__ == "__main__":
    print(f"OR(0,0)= {or_gate(0,0)}")
    print(f"OR(1,0)= {or_gate(1,0)}")
    print(f"OR(0,1)= {or_gate(0,1)}")
    print(f"OR(1,1)= {or_gate(1,1)}")
