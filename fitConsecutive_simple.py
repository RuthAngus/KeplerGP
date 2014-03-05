# GP deflation

import numpy as np
import pylab
import load_data
import GaussianProcesses as GP
# import Toeplitz_matrix_code as Tmc

def fitGP(time, flux):

    # Load data
    # time, flux = load_data.load('012317678', quarter = 3)
    # print len(time)

    # Observed data
    x1 = time - time[0]
    X = np.matrix([x1,]).T #convert to data matrix
    f = flux
    f -= f.mean() #mean centre the data

    # Predictive
    x1_pred = np.linspace(x1[0]-1, x1[-1]+1, len(x1))
    X_pred = np.matrix([x1_pred,]).T #convert to data matrix

    pylab.close(2)
    pylab.figure(2)
    
    #create the GP class
    MyGP = GP.GP(f,X,X_pred=X_pred,kernel='SqExp')
    MyGP.describe()

    #set the hyperparameters
    MyGP.set_attributes(hyperparams=[1.,0.5,0.4])

    # #get values for the predictive distribution *conditioned* on the observed data
    # f_pred, f_pred_err = MyGP.predict(X_pred,WhiteNoise=False)

    # #plot the data and regression
    # GP.PlotData(x1,f,MyGP.hyperparams[-1])
    # GP.PlotRanges(x1_pred,f_pred,f_pred_err,title="Initial hyperparams")
    # for i in range(5): #plot some random vectors
    #     pylab.plot(x1_pred,MyGP.GetRandomVector())

    ##########################################################################################
    #Learning the (hyper)parameters - ie parameters of the covariance matrix

    #optimise the hyperparameters instead of fixing them
    
    MyGP.MaxLikelihood()
    
    pylab.close(3)
    pylab.figure(3)
    f_pred, f_pred_err = MyGP.predict(X_pred,WhiteNoise=False)
    GP.PlotData(x1,f,MyGP.hyperparams[-1])
    GP.PlotRanges(x1_pred,f_pred,f_pred_err,title="Gaussian Process Regression")

    for i in range(5): #plot some random vectors
        pylab.plot(x1_pred,MyGP.GetRandomVector())
        
    detrended_flux = (f/f_pred) / np.median(f)
    rel_err = f_pred_err / np.median(f)
    # pylab.close(4)
    # pylab.figure(4)
    # # pylab.errorbar(x1[100:-100], detrended_flux[100:-100], yerr = rel_err[100:-100], fmt = 'k.')
    # pylab.plot(x1[100:-100], detrended_flux[100:-100], 'k.')

    return x1, detrended_flux
