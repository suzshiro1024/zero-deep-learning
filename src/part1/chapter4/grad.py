import numpy as np
from sample_function2 import func2
from numerical_gradient import numerical_gradient


def gradient_descent(f, init_x, lr, step_num=100):
    x = init_x

    for i in range(step_num):
        grad = numerical_gradient(f, x)

        x -= lr * grad
    return x


if __name__ == "__main__":
    init_x = np.array([-3.0, 4.0])

    print(f"result1 = {gradient_descent(func2, init_x=init_x, lr=0.1, step_num=100)}")
    print(f"result2 = {gradient_descent(func2, init_x=init_x, lr=10.0, step_num=100)}")
    print(f"result3 = {gradient_descent(func2, init_x=init_x, lr=1e-10, step_num=100)}")
