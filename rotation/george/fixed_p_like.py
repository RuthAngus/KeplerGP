import numpy as np
from scipy.linalg import cho_factor, cho_solve
import george
from george.kernels import ExpSine2Kernel, ExpSquaredKernel

def lnlike(theta, x, y, yerr):
    theta = np.exp(theta)
    k = theta[0]*ExpSine2Kernel(theta[1], theta[4]) * ExpSquaredKernel(theta[2])
    gp = george.GaussianProcess(k)
    gp.compute(x, (theta[3]*yerr**2))
    return gp.lnlikelihood(y)

def predict(xs, x, y, yerr, theta, P):
    theta = np.exp(theta)
    k = theta[0]*ExpSine2Kernel(theta[1], P) * ExpSquaredKernel(theta[2])
    gp = george.GaussianProcess(k)
    gp.compute(x, (theta[3]*yerr**2))
    return gp.predict(y, xs)

def neglnlike(theta, x, y, yerr, P):
    theta = np.exp(theta)
    k = theta[0]*ExpSine2Kernel(theta[1], P) * ExpSquaredKernel(theta[2])
    gp = george.GaussianProcess(k)
    gp.compute(x, (theta[3]*yerr**2))
    return -gp.lnlikelihood(y)

# I think theta[1] is 1/theta[1]**2
def QP(theta, x, yerr, P):
    r = x[:, None] - x[None, :]
    theta = np.exp(theta)
    K =  theta[0]*np.exp(-np.sin(r*np.pi/P)**2*theta[1])*\
            np.exp(-r**2/(2*theta[2]**2))
    K += np.diag(theta[3]*yerr**2)
    return K
