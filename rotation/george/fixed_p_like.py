import numpy as np
from scipy.linalg import cho_factor, cho_solve
import george
from george.kernels import ExpSine2Kernel, ExpSquaredKernel

def lnlike(theta, x, y, yerr, P):
    theta = np.exp(theta)
    k = theta[0]*ExpSine2Kernel(theta[1], P) * ExpSquaredKernel(theta[2])
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
