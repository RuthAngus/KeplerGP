# GP deflation

import numpy as np
import pylab
import load_data
import GaussianProcesses as GP
import marglike


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

    #--------------------------------------------------------------------
    
    # # Predictive
    # x1_pred = np.linspace(x1[0]-1, x1[-1]+1, len(x1))
    # X_pred = np.matrix([x1_pred,]).T # convert to data matrix

    # pylab.close(2)
    # pylab.figure(2)
    
    # # create the GP class
    # MyGP = GP.GP(f,X,X_pred=X_pred,kernel='SqExp')
    # MyGP.describe()

    # # set the hyperparameters
    # MyGP.set_attributes(hyperparams=[1.,0.5,0.4])

    # #get values for the predictive distribution *conditioned* on the observed data
    # f_pred, f_pred_err = MyGP.predict(X_pred,WhiteNoise=False)

    # #plot the data and regression
    # GP.PlotData(x1,f,MyGP.hyperparams[-1])
    # GP.PlotRanges(x1_pred,f_pred,f_pred_err,title="Initial hyperparams")
    # for i in range(5): #plot some random vectors
    #     pylab.plot(x1_pred,MyGP.GetRandomVector())

    # ##########################################################################################
    # #Learning the (hyper)parameters - ie parameters of the covariance matrix

    # #optimise the hyperparameters instead of fixing them
    # MyGP.MaxLikelihood()

    # pylab.close(3)
    # pylab.figure(3)
    # f_pred, f_pred_err = MyGP.predict(X_pred,WhiteNoise=False)
    # GP.PlotData(x1,f,MyGP.hyperparams[-1])
    # GP.PlotRanges(x1_pred,f_pred,f_pred_err,title="Gaussian Process Regression")

    # for i in range(5): #plot some random vectors
    #     pylab.plot(x1_pred,MyGP.GetRandomVector())
        
    # detrended_flux = (f/f_pred) / np.median(f)
    # rel_err = f_pred_err / np.median(f)
    # # pylab.close(4)
    # # pylab.figure(4)
    # # # pylab.errorbar(x1[100:-100], detrended_flux[100:-100], yerr = rel_err[100:-100], fmt = 'k.')
    # # pylab.plot(x1[100:-100], detrended_flux[100:-100], 'k.')

    # return x1, detrended_flux
