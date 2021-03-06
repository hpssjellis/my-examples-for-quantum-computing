import numpy as np

import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

fig = plt.figure(figsize=(5, 4), dpi=72)
axes = fig.add_axes([0.01, 0.01, .98, 0.98])
X = np.linspace(0, 2, 200)
Y = np.sin(2*np.pi*X)
plt.plot(X, Y, lw=2)
plt.ylim(-1.1, 1.1)
plt.grid()

plt.show()
