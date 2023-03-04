from sample_function1 import func1


def numerical_diff(f, x):
    h = 1e-4
    return (f(x + h) - f(x - h)) / (2 * h)


if __name__ == "__main__":
    print(f"f'(5) = {numerical_diff(func1, 5)}")
    print(f"f'(10) = {numerical_diff(func1, 10)}")
