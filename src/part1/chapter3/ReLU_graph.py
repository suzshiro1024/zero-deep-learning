import numpy as np
import matplotlib.pyplot as plt
import ReLU

if __name__ == "__main__":
    x = np.arange(-5.0, 5.0, 0.1)
    y = ReLU.relu(x)

    plt.plot(x, y)
    plt.ylim(-1, 5.1)

    plt.savefig("../../../figure/ReLU.png")
