# marglike returns the negative log-likelihood and its gradients w.r.t.
# the hyperparameters.

import numpy as np
import matrifysquare
from squared_exp import SE
from pcgToepPadding import pcgToepPadding
from matrifysquare import matrifysquare

def marglike(hyper, x, y, cadence):
    # FIXME: K and sophie cov function are not the same thing
    # FIXME: check matrix multiplication. A*B in matlab is np.dot(A, B) in python

    # Calculate covariance matrix
    K = matrifysquare(hyper, x, 0)
    K = 0.5* ( K + K.T) # Forces K to be perfectly symmetric

    # Calculate derivatives
    dKdsigma = matrifysquare(hyper, x, 1) # Derivative w.r.t. log(sigma)
    dkdlambda1 = matrifysquare(hyper, x, 2) # Derivative w.r.t. log(lambda1)
    dkdh1 = matrifysquare(hyper, x, 3) # Derivative w.r.t. log(h1)

    # Here's where I use Sophie's badass code!
    # invKy = K*-1*y
    # Cov_c = first column vector of cov matrix
    # Cov_r = first row vector of cov matrix
    # M = number of data files (1)
    # T = time vector
    # C = cadence vector
    # Y = flux vector
    # sv = array containing len(T), len(C), len(Y)
    # para = hyperparams
    M = 1
    T = x
    C = cadence
    Y = y
    sv = np.array([len(T), len(C), len(Y)])
    par = hyper
    
    #CovFunc = SE( M, T, C, Y, sv, par, False)
    Cov_c, Cov_r, inds, y = SE(M, T, C, Y, sv, par, True)
    alpha, flag, ii = pcgToepPadding(Cov_c, Cov_r, y, 10**(-6), 200, inds)
    raw_input('enter')
    

    U = np.linalg.cholesky(K)

    L = - sum(np.log(np.diag(U))) -0.5 * y * alpha - n*0.5*np.log(2*np.pi)
    dLdsigma = 0.5 * sum(np.diag(alpha*alpha.T*dkdsigma - (np.linalg.solve(K, dKdsigma)) ))
    dLdlambda1 = 0.5 * sum(np.diag(alpha*alpha.T*dkdlambda1 - (np.linalg.solve(K, dkdlambda1)) ))
    dkdh1 = 0.5 * sum(np.diag(alpha*alpha.T*dkdh1 - (np.linalg.solve(K, dkdh1)) ))

    return -L, [-dldsigma, -dLdlambda1, -dldh1]
