import numpy as np
import scipy.spatial as sp
import scipy.linalg as la
import scipy.optimize as so
from math import *
from scipy.io.matlab import mio
import time
from createToepCov import createMatCov
from detMatToepSym import detMatToepSym
from pcgToepPadding import pcgToepPadding
from detMatToepInc import detToepInc

data = mio.loadmat("q2_mod16_out4_raw.mat")
T2=data['time']
C2=data['cadence']
Y2=data['flux_arr_pdc']
N2=np.shape(T2)[0]

data = mio.loadmat("q3_mod24_out4_raw.mat")
T3=data['time']
C3=data['cadence']
Y3=data['flux_arr_pdc']
N3=np.shape(T3)[0]

data = mio.loadmat("q4_mod10_out4_raw.mat")
T4=data['time']
C4=data['cadence']
Y4=data['flux_arr_pdc']
N4=np.shape(T4)[0]

data = mio.loadmat("q5_mod02_out4_raw.mat")
T5=data['time']
C5=data['cadence']
Y5=data['flux_arr_pdc']
N5=np.shape(T5)[0]

M=2
T=np.vstack((T2,T3))
print np.shape(Y2)
C=np.vstack((C2,C3))
Y=np.hstack((Y2[0,:],Y3[0,:]))
sv=np.array([N2,N3]) #vector of size of each data

############################################################
# Compute a covariance matrix for a given kernel, given a #
# set of inputs and covariance paramters                   #           
############################################################

def SE(M,T,C,Y,sv,par,white_noise = False):
	# input
	# 	M number of data (here 4)
	# 	T=[T1,T2...]
	# 	C=[C1,C2...]
	# 	N=[N1,N2...] where N=size(T)=size(C)
	#	par parameter of the kernel function
    # Squared expodentiel * periodics
    if white_noise == True: # add white noise
        w=1
    else:
        w=0
    
    theta=par[0]
    P=par[1]
    l=par[2]
    L=par[3]
    sigma=par[4]
    f = lambda i,j,delta: theta * np.exp(-(np.sin(np.pi*(i-j)*delta/P))**2/(2*l**2) -(i-j)**2*delta**2/(2*(L)**2)) + w*(i==j)*sigma # f should depend of i,j,delta


    c,r,inds,y=createMatCov(M,T,C,Y,sv,f)
    return c,r,inds,y
	# output
	# 	c, r first row and first column of the covariance matrix
	# 	inds vector of the indice of hte real matrix

##################################################
# Functions relating to the GP regression itself #
##################################################

def NLL_GP(par, M,T,C,sv, Y, CovFunc):

    # Compute negative log likelihood for GP model for inputs X, outputs
    # y, covariance function CovFunc and covariance parameters par.
   
    # separation of the parameter
    c = par[0]
    para = par[1:]
    
    # create the covariance matrix and the vector y
    Cov_c,Cov_r,inds,y = CovFunc(M,T,C,Y,sv,para,True)
    print 'y = ', np.shape(y), 'inds = ', np.shape(inds)
    y=y[inds,:]
    # get log determinant of K
    print 'calcul determinant'
    # logdetK = np.log(detToepInc(Cov_c,inds))
    
    
    # print 'det calculer', logdetK
    # solve K x = y (via Toplitz solve) to get x = (K^-1)*y
    
    raw_input('enter')
    invKy, flag, ii = pcgToepPadding(Cov_c, Cov_r, y - c, 10**(-6), 200, inds)
    if flag == 1 :
        print 'error: pcg algorithm has not converge'
        exit()
    elif flag == 2 :	
        print 'error: '
        exit()
    elif flag == 3 :	
        print 'error: stagnation of pcg '
        exit()
    elif flag == 4 :	
        print 'error: some quantities became too small or too big for finish the compute'
        exit()
    elif flag == 0 :
        print 'pcg sucess fin inversion'
    print invKy
    logL = -0.5 * np.dot((y-c).T,invKy) - 0.5 * logdetK - (y.size/2.) * np.log(2*np.pi)
    print logL.shape
    
    return -logL

#################
# Example usage #
#################

y=np.vstack((Y.T))
CovFunc = SE
par_init = np.array([5.,10.,2.,10.,10.,1.])
print "Initial covariance parameters: ", par_init,"\n"
print "Initial NLL: ", NLL_GP(par_init, M,T,C,sv, y, CovFunc),"\n"
print 'Debut minimisation'
par = so.fmin(NLL_GP, par_init, (M,T,C,sv, y, CovFunc))
print "Maximum likelihood covariance parameters: ", par,"\n"
print "Final NLL: ",  NLL_GP(par, M,T,C,sv, y, CovFunc),"\n"




