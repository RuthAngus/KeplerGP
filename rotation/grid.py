# Specific version for rotation measurement.

import numpy as np
import pyfits
import scipy.spatial as sp
import scipy.linalg as la
import scipy.optimize as so
import pylab as pl
import emcee
import triangle

plotpar = {'axes.labelsize': 16,
           'text.fontsize': 16,
           'legend.fontsize': 14,
           'xtick.labelsize': 17,
           'ytick.labelsize': 17,
           'text.usetex': True}
pl.rcParams.update(plotpar)

def cf(X1, X2, theta, wn = False):
    D = sp.distance.cdist(X1, X2, 'sqeuclidean')
    K = theta[0]*np.exp(-D/(2*theta[1]**2) \
        -.5*np.sin(np.pi*np.sqrt(D)/theta[2])**2/theta[3]**2)
    if wn == True:
        K += np.identity(X1[:,0].size)*theta[-1]**2
    return np.matrix(K)

def predict(Xs, X, y, CovFunc, par, WhiteNoise = True, ReturnCov = False):
    K = CovFunc(X, X, par, wn = True) # training points
    Kss = CovFunc(Xs, Xs, par, wn = WhiteNoise) # test points
    Ks = CovFunc(Xs, X, par, wn = False) # cross-terms
    Kinv = np.linalg.inv( np.matrix(K) )
    y = np.matrix(np.array(y).flatten()).T
    prec_mean = Ks * Kinv * y
    prec_cov = Kss - Ks * Kinv * Ks.T
    if ReturnCov: # full covariance, or
        return np.array(prec_mean).flatten(), np.array(prec_cov)
    else: # just standard deviation
        return np.array(prec_mean).flatten(), np.array(np.sqrt(np.diag(prec_cov)))

# Load data
hdulist = pyfits.open("/Users/angusr/angusr/data2/Q3_public/kplr003223000-2009350155506_llc.fits")
tbdata = hdulist[1].data
x = tbdata["TIME"]
y = tbdata["PDCSAP_FLUX"]
yerr = tbdata["PDCSAP_FLUX_ERR"]

# remove nans
n = np.isfinite(x)*np.isfinite(y)*np.isfinite(yerr)
l = 500.
x = x[n][:l]
y = y[n][:l]
yerr = yerr[n][:l]

# subsample
xplot = x
yplot = y/np.mean(y) - 1
yerrplot = yerr
subsamp = 5
x = x[0:-1:subsamp]
y = y[0:-1:subsamp]
yerr = yerr[0:-1:subsamp]

# test data
xs = np.r_[min(x)-1:max(x)+1:500j]

# format data
y = np.array(y)/np.mean(y) - 1.
X = np.matrix([x]).T # convert inputs to matrix form (N x D)
Xs = np.matrix([xs]).T  # convert inputs to matrix form (Q x D)

# amplitude, squared exp length-scale (decay), period, periodic decay, white noise
theta = [1.,1.,2.,.4,.3] # quasi-periodic
covfunc = cf

def lnlike(h, X, y, covfunc):
    y = np.matrix(np.array(y).flatten()).T
    K = covfunc(X, X, h, wn = True)
    sign, logdetK = np.linalg.slogdet(K)
    alpha = np.mat(la.lu_solve(la.lu_factor(K),y))
    logL = -0.5*y.T * alpha - 0.5*logdetK - (y.size/2.)*np.log(2)
    return np.array(logL).flatten()

# uniform priors (quasi-periodic)
def lnprior(h):
    if 0.<h[0]<100. and 0.<h[1]<100. and 0.<h[2]<100. and 0.<h[3]<10. and 0.<h[4]<2.:
        return 0.0
    return -np.inf

# posterior prob
def lnprob(h, X, y, covfunc):
    lp = lnprior(h)
    if not np.isfinite(lp):
        return -np.inf
    return lp + lnlike(h, X, y, covfunc)

print "Initial parameters = ", theta
print "Initial lnlike = ", lnlike(theta,X,y,covfunc),"\n"

# compute prediction for initial guess and plot
print "plotting guess"
ys, ys_err = predict(Xs, X, y,covfunc, theta)
pl.clf()
pl.errorbar(xplot, yplot, color='k', fmt='.', capsize=0, ecolor='.7', zorder=2, alpha=.8)
pl.plot(xs, ys, color = "#339999", zorder=1, linewidth='2')
pl.xlabel("$\mathrm{Time~(days)}$")
pl.xlim(259, 272)
pl.savefig('guess')

# grid in P
P = np.arange(0.1, 10, 0.1)
lhs = np.zeros(len(P))

for i in range(len(P)):
    theta[2] = P[i]
    lhs[i] = 10**lnlike(theta, X, y, covfunc)

pl.clf()
pl.plot(P, lhs, 'k-')
pl.savefig('gridsearch')

print 'max likelihood period = ', P[lhs == max(lhs)]
print 'likelihood = ', max(lhs)

# Autocorrelation
# lags, acf, lines, axis = pl.acorr(y, maxlags = 20.)

# Periodogram
# f, Power = scipy.signal.periodogram(
