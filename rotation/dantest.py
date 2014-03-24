import numpy as np
import matplotlib.pyplot as pl
from lnlikefn import QP, lnlike

x = np.sort(np.random.rand(100))*10.
yerr = .01*np.ones_like(x)

theta = [1., 1., 1., 5.]
K = QP(x, x, theta) + np.diag(yerr**2)

pl.clf()
pl.imshow(K, cmap = 'gray', interpolation = 'nearest')
pl.savefig('K')

y = np.random.multivariate_normal(np.zeros_like(x), K)
pl.clf()
pl.errorbar(x, y, yerr = yerr, fmt = 'k.')
pl.plot(x, y, 'b-')
pl.savefig('xy')

P = np.arange(0.1, 3, 0.01)
L = np.empty_like(P)

theta2 = np.array(theta)
for i, per in enumerate(P):
    theta2[1] = per
    L[i] = lnlike(theta2, x, y, yerr)

pl.clf()
pl.plot(P, L, 'k-')
pl.savefig('likelihood')
