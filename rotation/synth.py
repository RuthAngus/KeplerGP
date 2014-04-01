import numpy as np
from lnlikefn import QP

def synthetic_data(x, yerr, theta):
    K = QP(x, x, yerr, theta)
    return np.random.multivariate_normal(np.zeros(len(x)), K)
