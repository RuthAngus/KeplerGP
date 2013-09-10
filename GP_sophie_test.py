import numpy as np
import scipy.spatial as sp
import scipy.linalg as la
import scipy.optimize as so
import pylab as plt
import load_data
import time
from pcgToepPadding import pcgToepPadding
from squared_exp_test import Squared_Exp
# from matrifysquare import matrifysquare


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
    # get log eterminant of K
    sign, logdetK = np.linalg.slogdet(K)
    # solve K x = y (via LU decomposition) to get x = (K^-1)*y
    logL = -0.5 * y.T *  np.mat(la.lu_solve(la.lu_factor(K),y)) \
        - 0.5 * logdetK - (y.size/2.) * np.log(2*np.pi)
    return -np.array(logL).flatten()

def PrecD_GP(Xs, X, y, cadence, CovFunc, par, WhiteNoise = True, ReturnCov = False):
    
    # evaluate covariance matrix, K
    Cov_c, Cov_r, inds, y = Squared_Exp(2, X, cadence, y, np.array([len(X)/2, len(X)/2]), par, False)
    K = la.toeplitz(Cov_c, Cov_r)[inds, inds]
    
    # Css, Rss, indsss, yss = \
        # Squared_Exp(2, Xs, cadence, y, np.array([len(Xs)/2, len(Xs)/2]), par, False)
    # Kss = la.toeplitz(Css, Rss)

    Xs_split = np.split(Xs, 2)
    X_split = np.split(X, 2)
    Xs_split = np.r_[Xs_split[0], Xs_split[1]]
    X_split = np.r_[X_split[0], X_split[1]]
    Xs_split = np.reshape(Xs_split, (len(Xs_split), 1)) # shape should be, eg (100, 1)
    X_split = np.reshape(X_split, (len(X_split), 1))
    K = CovFunc(X, X, par, white_noise = True) # training points
    Kss = CovFunc(Xs_split, Xs_split, par, white_noise = WhiteNoise) # test points
    Ks = CovFunc(Xs_split, X_split, par, white_noise = False) # cross-terms
    
    maxit = 200
    tolerance = 10**(-6)
    y=y[inds,:]
    # find K^-1*y
    strt = time.time()
    invKy, flag, ii = pcgToepPadding(Cov_c, Cov_r, y, tolerance, maxit, inds)
    print 'TIME = ', time.time() - strt
    
    # Kinv = np.linalg.inv( np.matrix(K) )
    
    # ensure y is an (n x 1) column matrix
    y = np.matrix(np.array(y).flatten()).T
    
    # evaluate mean and covariance of predictive distribution    
    prec_mean = np.dot(Ks, invKy) # (200, 1)
    
    # Kinv = invKy / y
    # print 'Kss', np.shape(Kss), 'Ks', np.shape(Ks), 'kinv', np.shape(Kinv), 'Ks.T', np.shape(Ks.T)
    # prec_cov = Kss - Ks * Kinv * Ks.T # <<< slow
    # prec_cov = Kss - Ks * Ks.T * Kinv 
        
    # return predictive values and either
    if ReturnCov: # full covariance, or
        return np.array(prec_mean).flatten(), np.array(prec_cov)
    else: # just standard deviation
        return np.array(prec_mean).flatten()#, np.array(np.sqrt(np.diag(prec_cov)))
    
# Load data Q2
time1, flux1, yerr1, cadence1 = load_data.load('012317678', quarter = 3, return_cadence = True)
# Load data Q3
time2, flux2, yerr2, cadence2 = load_data.load('012317678', quarter = 4, return_cadence = True)

# x1 = time1[:100]; x2 = time1[200:300]
# y1 = flux1[:100]; y2 = flux1[200:300]
# c1 = cadence1[:100]; c2 = cadence1[200:300]
length = 500
x1 = time1[:length]; x2 = time2[:length]
y1 = flux1[:length]; y2 = flux2[:length]
c1 = cadence1[:length]; c2 = cadence2[:length]

# Adjust flux 3 so that it follows on from flux 2 and mean centre
y2 -= y1[-1]
y1 -= np.mean(y1)
y2 -= np.mean(y2)

# Predictive data
step1 = (x1[-1] - x1[0]) / (len(x1))
step2 = (x2[-1] - x2[0]) / (len(x2))
xp1 = np.arange( x1[0], x1[-1], step1 ) # FIXME: check this
xp2 = np.arange( x2[0], x2[-1], step2 )
Xp1 = np.matrix([xp1]).T
Xp2 = np.matrix([xp2]).T

x = np.r_[x1, x2]
y = np.r_[y1, y2]
xp = np.r_[xp1, xp2]
X = np.matrix([x]).T # convert inputs to matrix form (N x D)

# CovFunc = SE
CovFunc = matrify
par_init = [1.,3,0.3]
   
print "Initial covariance parameters: ", par_init,
print "Initial NLL: ", NLL_GP(par_init, X, y, CovFunc),
#par = so.fmin(NLL_GP, par_init, (X, y, CovFunc))
par = par_init
print "Maximum likelihood covariance parameters: ", par
print "Final NLL: ", NLL_GP(par, X, y, CovFunc)

# Reshape (required for stacking) and stack data
x1 = np.reshape(x1, (len(x1), 1)); x2 = np.reshape(x2, (len(x2), 1))
y1 = np.reshape(y1, (len(y1), 1)); y2 = np.reshape(y2, (len(y2), 1))
c1 = np.reshape(c1, (len(c1), 1)); c2 = np.reshape(c2, (len(c2), 1))
xp1 = np.reshape(xp1, (len(xp1), 1)); xp2 = np.reshape(xp2, (len(xp2), 1))

x_stack = np.vstack((x1, x2))
y_stack = np.vstack((y1, y2))
c_stack = np.vstack((c1, c2))
xp_stack = np.vstack((xp1, xp2))

# evaluate predictive mean and standard deviation for the optimized parameters
# yp, yp_err = PrecD_GP(xp_stack, x_stack, y_stack, c_stack, CovFunc, par)
yp = PrecD_GP(xp_stack, x_stack, y_stack, c_stack, CovFunc, par)
# print 'yp_err = ', yp_err
# print len(yp)
yp += max(yp); y += max(y)
yp /= np.mean(yp); y /= np.mean(y)
yp -= 1; y -=1
plt.plot(xp, yp, 'r-')
plt.plot(X, y, 'k.')
# plt.plot(X, y, 'k.')
# plt.plot(xp, yp, 'b-')
# plt.plot(xp, yp+yp_err, 'b--')
# plt.plot(xp, yp-yp_err, 'b--')
plt.show()

