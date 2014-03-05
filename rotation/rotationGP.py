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

# quasi-periodic covariance kernel
# def QP(X, y, theta, white_noise = False):
def QP(X1, X2, theta, white_noise = False):
    D = sp.distance.cdist(X1,X2, "sqeuclidean")
#     Dscaled = np.pi * D * theta[1]
#     K = theta[0]**2*np.exp(-np.sin(Dscaled)**2/(2*(theta[2]**2))-(D**2/(2*theta[3]**2)))
    K = theta[0]**2*np.exp(-D**2/(2*theta[1]**2) - 2*np.sin(np.pi*D)**2/theta[2]**2)
    if white_noise == True:
#         K += (np.identity(X[:,0].size) * (theta[4]**2))
        K += (np.identity(X[:,0].size) * (theta[3]**2))
    return np.matrix(K)

# squared exponential covariance kernel
def SE(X1, X2, theta, white_noise = False):
    d = sp.distance.cdist(X1, X2, 'sqeuclidean')
    K = theta[0]**2 * np.exp( - d/(2*(theta[1]**2)))
    if white_noise == True:
        K += (np.identity(X1[:,0].size) * (theta[2]**2))
    return np.matrix(K)

def predict(Xs, X, y, CovFunc, par, WhiteNoise = True, ReturnCov = False):
    K = CovFunc(X, X, par, white_noise = True) # training points
    Kss = CovFunc(Xs, Xs, par, white_noise = WhiteNoise) # test points
    Ks = CovFunc(Xs, X, par, white_noise = False) # cross-terms
    Kinv = np.linalg.inv( np.matrix(K) )
    y = np.matrix(np.array(y).flatten()).T
    prec_mean = Ks * Kinv * y
    prec_cov = Kss - Ks * Kinv * Ks.T
    print np.diag(prec_cov)
    if ReturnCov: # full covariance, or
        return np.array(prec_mean).flatten(), np.array(prec_cov)
    else: # just standard deviation
#         return np.array(prec_mean).flatten(), np.array(np.sqrt(np.diag(prec_cov)))
        return np.array(prec_mean).flatten(), np.array(np.sqrt(np.diag(np.sqrt(prec_cov**2))))

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
yplot = y
yplot -= np.mean(y)
yerrplot = yerr
subsamp = 5
x = x[0:-1:subsamp]
y = y[0:-1:subsamp]
yerr = yerr[0:-1:subsamp]

# test data
print len(x)
xs = np.r_[min(x)-.5:max(x)+.5:100j]

# format data
y = np.array(y)
y -= np.mean(y)
X = np.matrix([x]).T # convert inputs to matrix form (N x D)
Xs = np.matrix([xs]).T  # convert inputs to matrix form (Q x D)

# initial hyperparameters (SE)
# h_init = [1.,3.,0.3]
# covfunc = SE

# initial hyperparameters (quasi-periodic)
# Hyperparameters are amplitude frequency, periodic length scale,
# theta[0] - sqrt maximum covariance parameter - gives 1sigma prior dist size
# theta[1] - frequency
# theta[2] - length scale of periodic term
# theta[3] - length scale of multiplicative sq exp
# theta[4] - white noise standard deviation if white_noise=True

# hyperparams:
# amplitude, squared exp length-scale (decay), period, white noise
# h_init = [.3,.5,.3,.3,0.3]
h_init = [100.,.09,2.,.3]
covfunc = QP

def lnlike(h, X, y, covfunc):
    y = np.matrix(np.array(y).flatten()).T
    K = covfunc(X, X, h, white_noise = True)
    sign, logdetK = np.linalg.slogdet(K)
    logL = -0.5 * y.T *  np.mat(la.lu_solve(la.lu_factor(K),y)) \
        - 0.5 * logdetK - (y.size/2.) * np.log(2*np.pi)
    return np.array(logL).flatten()

# # Gaussian priors
# def lnprior(h):
# #     return -.5*(h[0]+.2)**2 -.5*(h[1]+.2)**2 -.5*(h[2]+.2)**2 \
# #             -.5*(h[3]+.2)*2 -.5*(h[4]+.2)*2
# #     return -.5*(h[0]+.01)**2 -.5*(h[1]+.01)**2 -.5*(h[2]+.01)**2
#     return -.5*(h[0]+50.)**2 -.5*(h[1]+1.)**2 -.5*(h[2]+.5)**2 \
#             -.5*(h[3]+.5)*2

# uniform priors
def lnprior(h):
    if 0.<h[0]<1000. and 0.<h[1]<2. and .5<h[2]<10. and .1<h[3]<1.:
        return 0.0
    return -np.inf

# posterior prob
def lnprob(h, X, y, covfunc):
    lp = lnprior(h)
    if not np.isfinite(lp):
        return -np.inf
    return lp + lnlike(h, X, y, covfunc)

print "Initial parameters = ", h_init
print "Initial lnlike = ", lnlike(h_init,X,y,covfunc),"\n"

# Maximum likelihood optimisation
# def NLL_GP(h, X, y, covfunc):
#     y = np.matrix(np.array(y).flatten()).T
#     K = covfunc(X, X, h, white_noise = True)
#     sign, logdetK = np.linalg.slogdet(K)
#     logL = -0.5 * y.T *  np.mat(la.lu_solve(la.lu_factor(K),y)) \
#         - 0.5 * logdetK - (y.size/2.) * np.log(2*np.pi)
#     return -np.array(logL).flatten()

# print "Initial NLL: ", NLL_GP(h_init,X,y,covfunc),"\n"
# par = so.fmin(NLL_GP, h_init, (X,y,covfunc))
# print "Maximum likelihood covariance parameters: ", par,"\n"
# print "Final NLL: ", NLL_GP(par,X,y,covfunc),"\n"

# compute prediction for initial guess and plot
print "plotting guess"
ys, ys_err = predict(Xs, X, y,covfunc, h_init)
print ys_err
pl.clf()
pl.errorbar(xplot, yplot, color='k', fmt='.', capsize=0, ecolor='.7', zorder=2)
pl.plot(xs, ys, color = "#339999", zorder=1, linewidth='2')
pl.plot(xs, ys+ys_err, color = "#339999", zorder=1, linewidth='2',alpha='.1')
pl.plot(xs, ys-ys_err, color = "#339999", zorder=1, linewidth='2', alpha='.1')
pl.xlabel("$\mathrm{Time (days)}$")
pl.ylabel("$\mathrm{Flux}$")
pl.savefig('guess')
raw_input('enter')

# Sample the posterior probability for m.
nwalkers, ndim = 32, len(h_init)
p0 = [h_init+1e-4*np.random.rand(ndim) for i in range(nwalkers)]
sampler = emcee.EnsembleSampler(nwalkers, ndim, lnprob, args = (X,y,covfunc))
print("Burn-in")
p0, lp, state = sampler.run_mcmc(p0, 100)
sampler.reset()
print("Production run")
sampler.run_mcmc(p0, 1000)

print("Making triangle plot")
fig_labels = ["$0$", "$1$", "$2$", "$3$", "$4$"]
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

h = np.array(mcmc_result)[:, 0]
print 'mcmc result', h

print "Final lnlike = ", lnlike(h, X, y, covfunc),"\n"
ys, ys_err = predict(Xs, X, y, covfunc, h)

# plot
pl.clf()
pl.plot(x, y, 'k.')
pl.plot(xs, ys, 'b-')
pl.savefig('mcmc_result')
