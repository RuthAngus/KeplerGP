'''Very basic GP regression routines'''

import numpy as np
import scipy.spatial as sp
import scipy.linalg as la
import scipy.optimize as so
import pylab as plt

########################################################################
# Functions to compute a covariance matrix for a given kernel, given a #
# set of inputs and covariance parameters - add your own if you want   #
########################################################################

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

##################################################
# Functions relating to the GP regression itself #
##################################################

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
    '''
    Evaluate mean and covariance (if ReturnCov is True) or standard
    deviation (if ReturnCov is False) of predictive distribution of GP
    for a set of test inputs Xs, given observed inputs X and outputs
    y, covariance function CovFunc and covariance parameters par.
    The WhiteNoise parameter controls whether the white noise term is
    included when evaluating the variance of the predictive
    distribution.
    Currently, the inverse of the covariance matrix is evaluated using
    the numpy linalg.inv function.
    '''
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

#################
# Example usage #
#################

def test():
    '''Example showing how to use the other routines'''
    # create a fake set of training data
    x = np.array([-1.5,-1.,-0.75,-0.40,-0.25,0.00])
    X = np.matrix([x]).T # convert inputs to matrix form (N x D)
    y = np.array([-1.6,-1.3,-0.40, 0.10, 0.5 ,0.7])
    y -= y.mean() # mean-centre the data
    # plot the data
    plt.clf()
    plt.plot(x, y, 'ko')
    # a set of test data where a prediction is desired
    xp = np.r_[-5:5:101j] # from -5 to 5 with 101 points including endpoints
    Xp = np.matrix([xp]).T  # convert inputs to matrix form (Q x D)
    # select covariance function and initual guess for hyperparameters
    CovFunc = SE
    par_init = [1.,3,0.3]
    # evaluate predictive mean and standard deviation for the initial guess
    yp, yp_err = PrecD_GP(Xp, X, y, CovFunc, par_init)
    plt.plot(xp, yp, 'r-')
    plt.plot(xp, yp+yp_err, 'r--')
    plt.plot(xp, yp-yp_err, 'r--')
    # optimize the negative log likelihood wrt the covariance parameters
    # fmin is Nelder-Mead optimizer
    print "Initial covariance parameters: ", par_init,"\n"
    print "Initial NLL: ", NLL_GP(par_init,X,y,CovFunc),"\n"
    par = so.fmin(NLL_GP, par_init, (X,y,CovFunc))
    print "Maximum likelihood covariance parameters: ", par,"\n"
    print "Final NLL: ", NLL_GP(par,X,y,CovFunc),"\n"
    # evaluate predictive mean and standard deviation for the optimized parameters
    yp, yp_err = PrecD_GP(Xp, X, y, CovFunc, par)
    plt.plot(xp, yp, 'b-')
    plt.plot(xp, yp+yp_err, 'b--')
    plt.plot(xp, yp-yp_err, 'b--')
    plt.show()
    return
