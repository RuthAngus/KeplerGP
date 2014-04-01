import numpy as np
from scipy.linalg import cho_factor, cho_solve

def QP(X1, X2, yerr, theta):
    r = X1[:, None] - X2[None, :]
    theta = np.exp(theta)
    K =  theta[0]*np.exp(-np.sin(r*np.pi/theta[1])**2/theta[2]**2)*\
            np.exp(-r**2/(2*theta[3]**2))
    print np.shape(K), np.shape(np.diag(theta[4]*yerr**2)), np.shape(yerr)
    K += np.diag(theta[4]*yerr**2)
    return K

def lnlike(theta, x, y, yerr):
    K = QP(x, x, yerr, theta)
    L, flag = cho_factor(K)
    logdet = 2*np.sum(np.log(np.diag(L)))
    alpha = cho_solve((L, flag), y)
    return -.5* (np.dot(y, alpha) + logdet)

def predict(xs, x, y, yerr, theta):
    K = QP(x, x, yerr, theta)
    Kss = QP(xs, xs, np.ones_like(xs)*np.mean(yerr)*theta[4], theta)
    Ks = QP(xs, x,  np.zeros_like(xs), theta)
    Kinv = np.linalg.inv( K )
    return np.dot(Ks, np.linalg.solve(K, y)), None
