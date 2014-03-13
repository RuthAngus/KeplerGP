import numpy as np
import matplotlib.pyplot as pl
import scipy.spatial as sp

def cf(X1, X2, theta, wn = False, ktype = 'se'):
    D = sp.distance.cdist(X1, X2, 'sqeuclidean')
    print D
    if ktype == 'se':
        K = theta[0]*np.exp(-D/(2*theta[1]**2))
    elif ktype == 'p':
#         K = theta[0] * np.exp(-2*np.sin(np.pi*D/theta[2])**2/theta[1]**2)
#         K = theta[0]*np.cos(2*np.pi*np.sqrt(D)/theta[2])
        K = theta[0] * np.exp(-2*np.sin(np.pi*np.sqrt(D)/theta[2])**2/theta[1]**2)
    elif ktype == 'qp':
        K = theta[0]*np.exp(-D/(2*theta[1]**2) \
                -.5*np.sin(np.pi*np.sqrt(D)/theta[2])**2/theta[3]**2)
#         K = theta[0]*np.exp(-D/(2*theta[1]**2) \
#                 -2*np.sin(np.pi*D/theta[2])**2/theta[3]**2)
#         K = theta[0]*np.cos(2*np.pi*np.sqrt(D)/theta[2])*np.exp(-D/(2*theta[1]**2))
    if wn == True:
        K += np.identity(X1[:,0].size)*theta[-1]**2
    print K
    return np.matrix(K)

x = np.arange(0, 15., .1)
X = np.matrix(x).T
theta = [1., 1., 0.3] #se
# theta = [1., 3., 2., 0.3] # p
# theta = [1., 1., 1., 1., 0.3] # qp
# theta = [1., 2., 2., .3] # qp2
# K = cf(X, X, theta, ktype = 'qp')
K = cf(X, X, theta, ktype = 'se')

pl.clf()
pl.imshow(K, interpolation = 'nearest', cmap = 'gray')
pl.savefig('K')

ocols = ['#FF9933','#66CCCC' , '#FF33CC', '#3399FF', '#CC0066', '#99CC99', '#9933FF', '#CC0000']
# plot draws from multivariate Gaussian
pl.clf()
K = cf(X, X, theta, ktype = 'se')
draw = np.random.multivariate_normal(np.zeros(len(x)), K)
pl.plot(x, draw, color = ocols[0])
draw = np.random.multivariate_normal(np.zeros(len(x)), K)
pl.plot(x, draw+4, color = ocols[1])
draw = np.random.multivariate_normal(np.zeros(len(x)), K)
pl.plot(x, draw+8, color = ocols[2])
draw = np.random.multivariate_normal(np.zeros(len(x)), K)
pl.plot(x, draw+12, color = ocols[3])
pl.savefig('draws')
