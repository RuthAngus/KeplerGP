import numpy as np
import pyfits
import scipy.spatial as sp
import scipy.linalg as la
import scipy.optimize as so
import pylab as pl
import emcee
import triangle

def QP(X, y, theta, white_noise = False):
    D = sp.distance.cdist(X,y, "sqeuclidean")
    Dscaled = np.pi * D * theta[1]
    K = theta[0]**2*np.exp(-np.sin(Dscaled)**2/(2*(theta[2]**2))-(D**2/(2*theta[3]**2)))
    if white_noise == True:
        K += (np.identity(X[:,0].size) * (theta[4]**2))
    return np.matrix(K)

def SE(X1, X2, theta, white_noise = False):
    d = sp.distance.cdist(X1, X2, 'sqeuclidean')
    K = theta[0]**2 * np.exp( - d/(2*(theta[1]**2)))
    if white_noise == True:
        K += (np.identity(X1[:,0].size) * (theta[2]**2))
    return np.matrix(K)

def NLL_GP(par, X, y, CovFunc):
    y = np.matrix(np.array(y).flatten()).T
    K = CovFunc(X, X, par, white_noise = True)
    sign, logdetK = np.linalg.slogdet(K)
    logL = -0.5 * y.T *  np.mat(la.lu_solve(la.lu_factor(K),y)) \
        - 0.5 * logdetK - (y.size/2.) * np.log(2*np.pi)
    return -np.array(logL).flatten()

def PrecD_GP(Xs, X, y, CovFunc, par, WhiteNoise = True, ReturnCov = False):
    K = CovFunc(X, X, par, white_noise = True) # training points
    Kss = CovFunc(Xs, Xs, par, white_noise = WhiteNoise) # test points
    Ks = CovFunc(Xs, X, par, white_noise = False) # cross-terms
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
l = 100
x = x[n][:l]
y = y[n][:l]
yerr = yerr[n][:l]

# subsample
cadence = 5
x = x[0:-1:cadence]
y = y[0:-1:cadence]
yerr = yerr[0:-1:cadence]

# test data
xs = np.r_[min(x)-.5:max(x)+.5:101j]

# format data
y = np.array(y)
y -= np.mean(y)
X = np.matrix([x]).T # convert inputs to matrix form (N x D)
Xs = np.matrix([xs]).T  # convert inputs to matrix form (Q x D)

# initial hyperparameters (SE)
# h_init = [1.,3.,0.3]

# initial hyperparameters (quasi-periodic)
# Hyperparameters are amplitude frequency, periodic length scale,
# theta[0] - sqrt maximum covariance parameter - gives 1sigma prior dist size
# theta[1] - frequency
# theta[2] - length scale of periodic term
# theta[3] - length scale of multiplicative sq exp
# theta[4] - white noise standard deviation if white_noise=True
h_init = [.3,.5,.3,.3,0.3]

# just data
pl.clf()
pl.plot(x, y, 'k.')
pl.savefig('data')

covfunc = QP
print "Initial parameters = ", h_init
print "Initial nll = ", NLL_GP(h_init,X,y,covfunc)
ys, ys_err = PrecD_GP(Xs, X, y,covfunc, h_init)

# plot
pl.clf()
pl.plot(x, y, 'k.')
pl.plot(xs, ys, 'r-')
pl.plot(xs, ys+ys_err, 'r--')
pl.plot(xs, ys-ys_err, 'r--')
pl.plot(xs, ys, 'b-')
pl.plot(xs, ys+ys_err, 'b--')
pl.plot(xs, ys-ys_err, 'b--')
pl.savefig('guess')

# Gaussian priors
def lnprior(h):
    return -.5*(h[0]+.2)**2 -.5*(h[1]+.2)**2 -.5*(h[2]+.2)**2 -.5*(h[3]+.2)**2

# posterior prob
def lnprob(h, X, t, covfunc):
    lp = lnprior(h)
    if not np.isfinite(lp):
        return -np.inf
    return lp + NLL_GP(h, X, y, covfunc)

# Sample the posterior probability for m.
nwalkers, ndim = 32, len(h_init)
p0 = [h_init+1e-4*np.random.rand(ndim) for i in range(nwalkers)]
sampler = emcee.EnsembleSampler(nwalkers, ndim, lnprob)
print("Burn-in")
p0, lp, state = sampler.run_mcmc(p0, 100)
sampler.reset()
print("Production run")
sampler.run_mcmc(p0, 500)

print("Making triangle plot")
fig_labels = ["$0$", "$1$", "$2$", "$3$"]
fig = triangle.corner(sampler.flatchain, truths=h_init, labels=fig_labels[:len(h_init)])
fig.savefig("triangle.png")

print("Plotting traces")
pl.figure()
for i in range(ndim):
    pl.clf()
    pl.axhline(h_init[i], color = "r")
    pl.plot(sampler.chain[:, :, i].T, 'k-', alpha=0.3)
    pl.savefig("{0}.png".format(i))

# Flatten chain
samples = sampler.chain[:, 50:, :].reshape((-1, ndim))

# Find values
mcmc_result = map(lambda v: (v[1], v[2]-v[1], v[1]-v[0]),
                  zip(*np.percentile(samples, [16, 50, 84], axis=0)))

mcmc_result = np.array(mcmc_result)[:, 0]
print 'mcmc result', mcmc_result

# h = so.fmin(NLL_GP, h_init, (X,y,covfunc))
# print "ml parameters = ", h,"\n"
# print "Final nll = ", NLL_GP(h,X,y,covfunc),"\n"
ys, ys_err = PrecD_GP(Xs, X, y,covfunc, mcmc_result)

# plot
pl.clf()
pl.plot(x, y, 'k.')
pl.plot(xs, ys, 'r-')
pl.plot(xs, ys+ys_err, 'r--')
pl.plot(xs, ys-ys_err, 'r--')
pl.plot(xs, ys, 'b-')
pl.plot(xs, ys+ys_err, 'b--')
pl.plot(xs, ys-ys_err, 'b--')
pl.savefig('gpfit')
