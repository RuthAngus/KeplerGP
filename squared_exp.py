
import numpy as np
from createToepCov import createMatCov

def SE(M, T, C, Y, sv, par, white_noise = False):
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
    
    # theta=par[0]
    # P=par[1]
    # l=par[2]
    # L=par[3]
    # sigma=par[4]
    # f = lambda i,j,delta: theta * \
    # 	np.exp(-(np.sin(np.pi*(i-j)*delta/P))**2/(2*l**2) \
    # 	       -(i-j)**2*delta**2/(2*(L)**2)) + w*(i==j)*sigma # f should depend of i,j,delta

    # Delta is the mean value?

    sigma = par[0]
    h1 = par[1]
    lambda1 = par[2]
    f = lambda i, j: h1**2 * np.exp(-(i-j)**2 / 2.*lambda1**2) # FIXME: this is bollocks
    
    c, r, inds, y = createMatCov(M, T, C, Y, sv, f)    
    return c, r, inds, y
	# output
	# 	c, r first row and first column of the covariance matrix
	# 	inds vector of the indice of hte real matrix

