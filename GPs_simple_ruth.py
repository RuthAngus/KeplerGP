import numpy as np
import scipy.spatial as sp
import scipy.linalg as la
import scipy.optimize as so
import pylab as plt
import load_data
from matrifysquare import matrifysquare


def SE(X1, X2, theta, white_noise = False):
    '''
    Squared exponential covariance function
    '''
    
    X1, X2 = np.matrix(X1), np.matrix(X2) # ensure both sets of inputs are matrices

    D2 = sp.distance.cdist(X1, X2, 'sqeuclidean') # calculate squared Euclidean distance

    K = theta[0]**2 * np.exp(- D2 / (2*(theta[1]**2))) # calculate covariance matrix
    if white_noise == True: # add white noise
        K += (np.identity(X1[:,0].size) * (theta[2]**2))
    return np.matrix(K)

#def matrifysquare(hyper, x1, x2, q, white_noise = False):
def matrifysquare(hyper, X1, X2, white_noise = False):
    n = len(X1) 
    sigma = np.exp(hyper[0])
    lambda1 = np.exp(hyper[1])
    h1 = np.exp(hyper[2])
    x1 = np.matrix(X1); X2 = np.matrix(X2) # ensure both sets of inputs are matrices
    diff = sp.distance.cdist(X1, X2, 'sqeuclidean') # calculate squared Euclidean distance
    # if q == 0:
    #     # covariance matrix
    #     K = h1**2 * np.exp(- diff / (2*(lambda1**2))) + sigma* np.matrix(np.identity(n))
    # elif q == 1:
    #     # derivative of K wrt hyper(1) = log(sigma)
    #     K = sigma * np.matrix(np.identity(n))
    # elif q == 2:
    #     # derivative of K wrt hyper(2) = log(lambda1)
    #     K = h1**2 * np.exp( - diff**2 / (2*lambda1**2)) * (diff**2/(lambda1**2))
    # elif q == 3:
    #     # derivative of K wrt hyper(3) = log(h1)
    #     K = 2* h1**2 * np.exp( - diff **2 / (2*lambda1**2))
    K = h1**2 * np.exp(- diff / (2*(lambda1**2))) + sigma* np.matrix(np.identity(n))
    return K

def matrify(X1, X2, hyper, white_noise = False):
    lambda1 = np.exp(hyper[0])
    h1 = np.exp(hyper[1])
    X1 = np.matrix(X1); X2 = np.matrix(X2) # ensure both sets of inputs are matrices
    diff = sp.distance.cdist(X1, X2, 'sqeuclidean') # calculate squared Euclidean distance
    K = h1**2 * np.exp(- diff / (2*(lambda1**2)))
    if white_noise == True: # add white noise
        K += (np.identity(X1[:,0].size) * (hyper[2]**2))
    return K

def marglike(x, y, hyper, white_noise = False): # FIXME: build optional white noise into this kernel

    # Calculate covariance matrix
    K = matrifysquare(hyper, x, 0)
    K = 0.5* ( K + K.T) # Forces K to be perfectly symmetric

    # Calculate derivatives
    # dKdsigma = matrifysquare(hyper, x, 1) # Derivative w.r.t. log(sigma)
    # dKdlambda1 = matrifysquare(hyper, x, 2) # Derivative w.r.t. log(lambda1)
    # dKdh1 = matrifysquare(hyper, x, 3) # Derivative w.r.t. log(h1)

    sign, logdetK = np.linalg.slogdet(K)
    
    invKy = -0.5 * y.T *  np.mat(la.lu_solve(la.lu_factor(K),y)) \
        - 0.5 * logdetK - (y.size/2.) * np.log(2*np.pi)
    
    U = np.linalg.cholesky(K)

    n = len(x)
    L = - sum(np.log(np.diag(U))) -0.5 * y * invKy - n*0.5*np.log(2*np.pi)
    # dLdsigma = 0.5 * sum(np.diag(invKy*invKy.T*dKdsigma - (np.linalg.solve(K, dKdsigma)) ))
    # dLdlambda1 = 0.5 * sum(np.diag(invKy*invKy.T*dKdlambda1 - (np.linalg.solve(K, dKdlambda1)) ))
    # dKdh1 = 0.5 * sum(np.diag(invKy*invKy.T*dKdh1 - (np.linalg.solve(K, dKdh1)) ))

    return -L #, [-dKdsigma, -dKdlambda1, -dKdh1]

def NLL_GP(par, X, y, CovFunc):
    '''
    Compute negative log likelihood for GP model for inputs X, outputs
    y, covariance function CovFunc and covariance parameters par.
    Currently uses LU decomposition to solve the linear system K*x=y
    '''
    # ensure y is an (n x 1) column vector
    y = np.matrix(np.array(y).flatten()).T
    # create the covariance matrix
    K = CovFunc(X, X, par, white_noise = True)
    # get log determinant of K
    sign, logdetK = np.linalg.slogdet(K)
    # solve K x = y (via LU decomposition) to get x = (K^-1)*y
    logL = -0.5 * y.T *  np.mat(la.lu_solve(la.lu_factor(K),y)) \
        - 0.5 * logdetK - (y.size/2.) * np.log(2*np.pi)
    return -np.array(logL).flatten()

def PrecD_GP(Xs, X, y, CovFunc, par, WhiteNoise = True, ReturnCov = False):
    # evaluate covariance matrices
    K = CovFunc(X, X, par, white_noise = True) # training points
    Kss = CovFunc(Xs, Xs, par, white_noise = WhiteNoise) # test points
    Ks = CovFunc(Xs, X, par, white_noise = False) # cross-terms
    # invert the covariance matrix for the training points
    Kinv = np.linalg.inv( np.matrix(K) )
    # ensure y is an (n x 1) column matrix
    y = np.matrix(np.array(y).flatten()).T
    # evaluate mean and covariance of predictive distribution
    prec_mean = Ks * Kinv * y
    prec_cov = Kss - Ks * Kinv * Ks.T
    # return predictive values and either
    if ReturnCov: # full covariance, or
        return np.array(prec_mean).flatten(), np.array(prec_cov)
    else: # just standard deviation
        return np.array(prec_mean).flatten(), np.array(np.sqrt(np.diag(prec_cov)))

# load data
time, flux, cadence = load_data.load('012317678', quarter = 3)   
x = time[0:100]
y = flux[0:100]

X = np.matrix([x]).T # convert inputs to matrix form (N x D)
y -= y.mean() # mean-centre the data
    
xp = np.r_[ x[0] : x[-1] : 200j ]
Xp = np.matrix([xp]).T  

# CovFunc = SE
CovFunc = matrify
# CovFunc1 = matrifysquare
# CovFunc2 = matrify
par_init = [1.,3,0.3]
# par_init = np.log([0.7**2, 600., 2.]) # same as Dona's

mu, sigma = PrecD_GP(Xp, X, y, CovFunc, par_init)
   
print "Initial covariance parameters: ", par_init,
print "Initial NLL: ", NLL_GP(par_init, X, y, CovFunc),
par = so.fmin(NLL_GP, par_init, (X, y, CovFunc))
print "Maximum likelihood covariance parameters: ", par
print "Final NLL: ", NLL_GP(par, X, y, CovFunc)

# evaluate predictive mean and standard deviation for the optimized parameters
yp, yp_err = PrecD_GP(Xp, X, y, CovFunc, par)
plt.plot(X, y, 'k.')
plt.plot(xp, yp, 'b-')
plt.plot(xp, yp+yp_err, 'b--')
plt.plot(xp, yp-yp_err, 'b--')
plt.show()

