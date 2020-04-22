import numpy as np
import pylab as plt
omega = 0.8  # 1.
var1 = 0.3  # 0.1
var2 = 0.5  # 0.7


def mog(y, f):
    # sin = np.sin(f-0.7*y) ** 2
    # return npdf(y, f, var1) * sin
    return npdf(y, f-omega, var1) + npdf(y, f+omega, var2)


def npdf(x, m, v):
    return np.exp(-(x - m) ** 2 / (2 * v)) / np.sqrt(2 * np.pi * v)


# print(np.random.binomial(1, .5, 100))
y = 0.
f = np.linspace(-5., 5., num=1000)
mog_f = mog(y, f)
plt.plot(f, mog_f)
plt.show()
