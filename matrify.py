# x1 and x2 need to be column vectors.

# Hyper = log[input scale lambda
#             output scale h
#             noise variance sigman2]

# matrify returns the covariance matrix K(x,x*)

import numpy as np

def matrify(hyper, x1, x2):
    lambda1 = np.exp(hyper[1])
    h1 = np.exp(hyper[2])
    dif = x1 - x2
    return h1**2 * np.exp( -0.5* (dif/lambda1) **2 )
    

