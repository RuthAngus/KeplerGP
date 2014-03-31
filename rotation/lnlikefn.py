import numpy as np
from scipy.linalg import cho_factor, cho_solve

def QP(X1, X2, theta):
    r = X1[:, None] - X2[None, :]
    # sample in log space
    theta = np.exp(theta)
    return theta[0]*np.exp(-np.sin(r*np.pi/theta[1])**2/theta[2]**2)*np.exp(-r**2/(2*theta[3]**2))

    # fix period
#     return theta[0]*np.exp(-np.sin(r*np.pi/1.)**2/theta[1]**2)*np.exp(-r**2/(2*theta[2]**2))

def lnlike(theta, x, y, yerr):
    K = QP(x, x, theta) + np.diag(yerr**2)
    L, flag = cho_factor(K)
    logdet = 2*np.sum(np.log(np.diag(L)))
    alpha = cho_solve((L, flag), y)
    return -.5* (np.dot(y, alpha) + logdet)

def predict(xs, x, y, yerr, theta):
    K = QP(x, x, theta) + np.diag(yerr**2)
    Kss = QP(xs, xs, theta)
    Ks = QP(xs, x, theta)
    Kinv = np.linalg.inv( K )
    return np.dot(Ks, np.linalg.solve(K, y)), None
