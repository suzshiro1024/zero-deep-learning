import numpy as np
import matplotlib.pyplot as plt
import step_2 as step

if __name__ == "__main__":
    x = np.arange(-5.0, 5.0, 0.1)
    y = step.step(x)

    plt.plot(x, y)
    plt.ylim(-0.1, 1.1)

    plt.savefig("../../../figure/step.png")
