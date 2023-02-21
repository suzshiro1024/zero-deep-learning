import numpy as np
import matplotlib.pyplot as plt
import sigmoid


if __name__ == "__main__":
    x = np.arange(-5.0, 5.0, 0.1)
    y = sigmoid.sigmoid(x)

    plt.plot(x, y)
    plt.ylim(-0.1, 1.1)

    plt.savefig("../../../figure/sigmoid.png")
