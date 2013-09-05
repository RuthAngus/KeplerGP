# x1 and x2 need to be column vectors.

# Hyper = log[input scale lambda
#             output scale h
#             noise variance sigman2]

# matrify returns the covariance matrix K(x,x*)

import numpy as np
import scipy.spatial as sp

def matrify(hyper, x1, x2):
    
    lambda1 = np.exp(hyper[1])
    h1 = np.exp(hyper[2])
    # diff = x1 - x2
    x1 = np.matrix(x1); x2 = np.matrix(x2) # ensure both sets of inputs are matrices
    diff = sp.distance.cdist(x1, x2, 'sqeuclidean') # calculate squared Euclidean distance

    return h1**2 * np.exp(- diff / (2*(lambda1**2)))
    # return h1**2 * np.exp( -0.5* (diff/lambda1) **2 )
    

