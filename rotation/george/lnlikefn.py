import numpy as np
from scipy.linalg import cho_factor, cho_solve
import george
from george.kernels import ExpSine2Kernel, ExpSquaredKernel

def QP(X1, X2, yerr, theta):
    r = X1[:, None] - X2[None, :]
    theta = np.exp(theta)
    K =  theta[0]*np.exp(-np.sin(r*np.pi/theta[1])**2/theta[2]**2)*\
            np.exp(-r**2/(2*theta[3]**2))
    j2 = np.exp(2)*theta[4]
    K += np.diag(np.sqrt(yerr**2+j2))
#     K += np.diag(theta[4]*yerr**2)
    return K

def lnlike(theta, x, y, yerr):
    theta = np.exp(theta)
    k = theta[0]*ExpSine2Kernel(theta[2], theta[1]) * ExpSquaredKernel(theta[3])
    gp = george.GaussianProcess(k)
    j2 = np.exp(2)*theta[4]
    gp.compute(x, np.sqrt(yerr**2 + j2))
    return gp.lnlikelihood(y)

def predict(xs, x, y, yerr, theta):
    theta = np.exp(theta)
    k = theta[0]*ExpSine2Kernel(theta[2], theta[1]) * ExpSquaredKernel(theta[3])
    gp = george.GaussianProcess(k)
    j2 = np.exp(2)*theta[4]
    gp.compute(x, np.sqrt(yerr**2 + j2))
    return gp.predict(y, xs)
