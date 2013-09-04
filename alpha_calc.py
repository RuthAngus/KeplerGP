# Calculate alpha

import numpy as np
from squared_exp import SE
from pgcToepPadding import pgcToepPadding

def alpha_calc(M, T, C, Y, sv, par):
    # invKy = K*-1*y
    # Cov_c = first column vector of cov matrix
    # Cov_r = first row vector of cov matrix
    # M = number of data files (1)
    # T = time vector
    # C = cadence vector
    # Y = flux vector
    # sv = array containing len(T), len(C), len(Y)
    # para = hyperparams
    
    # M = 2
    # T = x
    # C = cadence
    # Y = y
    # sv = np.array([N1,N2])
    # par = hyper

    Cov_c, Cov_r, inds, y = SE(M, T, C, Y, sv, par, False)
    maxit = 200
    tolerance = 10**(-6)
    y=y[inds,:]
    alpha, flag, ii = pcgToepPadding(Cov_c, Cov_r, y, tolerance, maxit, inds)
