import matplotlib.pyplot as plt
import numpy as np

x = np.arange(0, 10, 1)
nu_min = 50
nu_max = 100
T_max = 10
y1 = nu_min + 1 / 2 * (nu_max - nu_min) * (1 + np.cos(np.pi * x / T_max))

plt.scatter(x, y1)
plt.savefig("plot.png")
