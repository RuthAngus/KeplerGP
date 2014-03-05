# GP deflation

import numpy as np
import pylab as p
import load_data
import GaussianProcesses as GP
import marglike
import fitGauss

def fitGP():

    # Load data Q2
    time1, flux1, cadence1 = load_data.load('012317678', quarter = 3)
    # Load data Q3
    time2, flux2, cadence2 = load_data.load('012317678', quarter = 4)
    
    # Adjust flux 3 so that it follows on from flux 2
    flux2 -= flux1[-1]
    
    # centre data about 0
    flux1 -= np.mean(flux1)
    flux2 -= np.mean(flux2)
   
    subsamp = 37
    subsamp2 = 17

    # subsample from data
    y1 = flux1[0:-1:subsamp]
    y2 = flux2[0:-1:subsamp]
    x1 = time1[0:-1:subsamp]
    x2 = time2[0:-1:subsamp]
    c1 = cadence1[0:-1:subsamp]
    c2 = cadence2[0:-1:subsamp]

    # set initial hyperparameters
    hyper = np.log([0.7**2, 600., 2.]) # same as Dona's

    # Stack data
    # N1 = np.shape(x1)[0]
    # N2 = np.shape(x2)[0]
    x1 = np.reshape(x1, (len(x1), 1)); x2 = np.reshape(x2, (len(x2), 1))
    y1 = np.reshape(y1, (len(y1), 1)); y2 = np.reshape(y2, (len(y2), 1))
    c1 = np.reshape(c1, (len(c1), 1)); c2 = np.reshape(c2, (len(c2), 1))
    x1 = np.vstack((x1, x2))
    # y1 = np.hstack((y1[0,:], y2[0,:]))
    y1 = np.vstack((y1, y2))
    cadence = np.vstack((c1, c2))
    # sv=np.array([N1, N2]) #vector of size of each data

    # minimise NLL
    print 'Minimising NLL...'
    marglike.marglike(hyper, x1, y1, cadence)

    # optimise hyperparams: hyper, fval = fmin(marglike(hyper11,x1,y1), hyperinit1)

    # [Mu11, Sigma11] = fitGauss(x1,y1,xstar1,hyper11);
    
    # xstar is the training data
    xstar = np.r_[ x1[0] : x1[-1] : 200j ] # FIXME: - not sure what the best option for xstar is
    mu, sigma = fitGauss.fitGauss(hyper, x1, y1, xstar, cadence)

    p.close(1)
    p.figure(1)
    p.plot(time1, flux1, 'k.')
    p.plot(time2, flux2, 'k.')
