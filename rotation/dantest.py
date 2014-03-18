import numpy as np
import matplotlib.pyplot as pl
from scipy.linalg import cho_factor, cho_solve

def QP(X1, X2, theta):
    r = X1[:, None] - X2[None, :]
    return theta[0]*np.exp(-np.sin(r*np.pi/theta[1])**2/theta[2]**2)*np.exp(-r**2/(2*theta[3]**2))

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

def lnlike(theta, x, y, yerr):
    K = QP(x, x, theta) + np.diag(yerr**2)
    L, flag = cho_factor(K)
    logdet = 2*np.sum(np.log(np.diag(L)))
    alpha = cho_solve((L, flag), y)
    return -.5* (np.dot(y, alpha) + logdet)

P = np.arange(0.1, 3, 0.01)
L = np.empty_like(P)

theta2 = np.array(theta)
for i, per in enumerate(P):
    theta2[1] = per
    L[i] = lnlike(theta2, x, y, yerr)

pl.clf()
pl.plot(P, L, 'k-')
pl.savefig('likelihood')
