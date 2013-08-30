# x needs to be a column vector.

# If q ==0, matrifysquare returns a square covariance matrix K(x,x)

# If q == 1, the output is the derivative w.r.t. Hyper(1), i.e.
# dK/d(log(Sigman2)) = dK/dSigman2*dSigman2/d(log(sigman2))

# If q == 2, matrifysquare returns the derivative of K(x,x) w.r.t.
# hyper(2), i.e. dK/d(log(lambda1)) = dK/dlambda1*dlambda1/d(log(lambda1))

# If q == 3, matrifysquare returns the derivative of K w.r.t the Hyper(3)
# i.e. dK/d(log(h1))= dK/dh1*dh1/d(log(h1))

# The kernels are:
# SQDEXP: k = h2^2*exp(-(x1-x2)^2/2*lambda2^2)

# hyper is an array of hyperparams

import numpy as np

def matrifysquare(hyper, x, q):

    n = len(x)-1 # FIXME: take 1 away so I is same size as x
    sigma = np.exp(hyper[0])
    lambda1 = np.exp(hyper[1])
    h1 = np.exp(hyper[2])

    # shift elements of x
    x2 = x[1:]
    x = x[:-1]
    diff = x - x2 # FIXME: is this the right thing to do? sort it out!

    if q == 0:
        # covariance matrix
        K = h1**2 * np.exp( - (diff**2) / (2*lambda1**2)) + sigma* np.matrix(np.identity(n))

    elif q == 1:
        # derivative of K wrt hyper(1) = log(sigma)
        K = sigma * np.matrix(np.identity(n))

    elif q == 2:
        # derivative of K wrt hyper(2) = log(lambda1)
        K = h1**2 * np.exp( - diff**2 / (2*lambda1**2)) * (diff**2/(lambda1**2))

    elif q == 3:
        # derivative of K wrt hyper(3) = log(h1)
        K = 2* h1**2 * np.exp( - diff **2 / (2*lambda1**2))

    return K
