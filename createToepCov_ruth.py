#!/Applications/Python3.3


import numpy as np
from numpy.fft import *
from math import *
from scipy.io.matlab import mio
from pdtMatToepVectPadding import pdtMatToepVect
from pcgToepPadding import pcgToepPadding
import time

def createToepCov(M,T,C,Y,sv):
# create the toeplitz matrix we need
# M number of different array you have
# T array which contain all the time vector T=vstack((T0,T1,T2 ...))
# C array which contain all the time vector C=vstack((C0,C1,C2 ...))
# sv vector which contain the lenght of vector C and T, N[0]=size(T0),size(C0)...
    global n
    n = 0 # mean value of the time gap
    d = np.zeros((M,1))
    for i in range(0,M):
        Te = T[n:n+sv[i],:] 
        Ce = C[n:n+sv[i],:]
        Ye = Y[n:n+sv[i],:]
        # Remove nanas
        T_nan = np.isnan(Te)
        Y_nan = np.isnan(Ye)
        ind1,j=np.where(T_nan==False)
        ind2,j=np.where(Y_nan==False)
        # ind1 and ind2 are indices of nans
        if i==0 :
            # For the first array (q2), Cf is the q2 cadence
            Cf=Ce
            Yf=Ye
            N=np.shape(Cf)[0]
            inds1=ind1
            inds2=ind2
        else:
            # k is the first cadence in q2 minus the last in the last cadence array - 1
            k = Ce[0] - Cf[-1] - 1
            Cf = np.vstack( (Cf, np.vstack(np.r_[ (Cf[-1] + 1):(Ce[0]) ])) )

            N=np.shape(Cf)[0]
            inds1=np.hstack((inds1,N+ind1))
            
            inds2=np.hstack((inds2,N+ind2))
            
            Cf=np.vstack((Cf,Ce))
            Yf=np.vstack((Yf,np.zeros((k,1)),Ye))
        n=n+sv[i];

        # mean value
        Te=Te[ind1]
        d[i]=np.mean(Te[1:]-Te[0:-1])
    
    delta=np.mean(d)

    # Intersect inds1 and inds2
    inds=np.intersect1d(inds1,inds2)

    # return Cf,inds,delta,Yf
    return Cf, inds, Yf

def createMatCov(M, T, C, Y, sv, f):
# c,r first column and first row of the covariance matrix
# inds vector who contain the indice we need for make the multiplication...
    
# M number of different array you have
# T array which contain all the time vector T=vstack((T0,T1,T2 ...))
# C array which contain all the time vector C=vstack((C0,C1,C2 ...))
# sv vector which contain the lenght of vector C and T, N[0]=size(T0),size(C0)...
# f kernel !!f should work with vector!!

    # Cf, inds, delta, Yf = createToepCov(M, T, C, Y, sv)
    Cf, inds, Yf = createToepCov(M, T, C, Y, sv)
    
    # c = f(Cf, Cf[0], delta)
    # print Cf
    # print Cf[0]
    # print len(Cf), type(Cf), np.shape(Cf)
    # print len(Cf[0]), type(Cf[0]), np.shape(Cf[0])
    c = f(Cf, Cf[0])
    r = c.T
    return c, r, inds, Yf
    
# for make a multiplication K*x used pdtMatToepVect(c,r,x,inds) 
    # x should have the size of inds
# for make a linear system resolution K*y=x used pcgToepPadding(c,r,x,tol,maxit,inds)
    # x should have the size of inds
    
