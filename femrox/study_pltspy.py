import matplotlib.pyplot as plt
import numpy as np

#https://matplotlib.org/stable/api/_as_gen/matplotlib.pyplot.spy.html

# Fixing random state for reproducibility
# np.random.seed(19680801)

# fig, axs = plt.subplots(2, 2)
# ax1 = axs[0, 0]
# ax2 = axs[0, 1]
# ax3 = axs[1, 0]
# ax4 = axs[1, 1]

# x = np.random.randn(20, 20)
# x[5, :] = 0.
# x[:, 12] = 0.

# ax1.spy(x, markersize=5)
# ax2.spy(x, precision=0.1, markersize=5)

# ax3.spy(x)
# ax4.spy(x, precision=0.1)

# plt.show()

a = np.arange(400).reshape(20,20)
idx = (1,3,6,7)
idx2d = np.ix_(idx,idx)
a[idx2d] = 0
a[10] = 0
a[:,10] = 0
plt.spy(a)
plt.show()
