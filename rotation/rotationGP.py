import numpy as np
import pyfits
import scipy.spatial as sp
import scipy.linalg as la
import scipy.optimize as so
import pylab as pl
import emcee
import triangle
# import GPSuz
# import GPSuz.GP_covmat as cf

plotpar = {'axes.labelsize': 16,
           'text.fontsize': 16,
           'legend.fontsize': 14,
           'xtick.labelsize': 17,
           'ytick.labelsize': 17,
           'text.usetex': True}
pl.rcParams.update(plotpar)

def cf(X1, X2, theta, wn = False, ktype = 'qp'):
    D = sp.distance.cdist(X1, X2, 'sqeuclidean')
    if ktype == 'se':
        K = theta[0]*np.exp(-D/(2*theta[1]**2))
    elif ktype == 'p':
        K = theta[0] * np.exp(-2*np.sin(np.pi*np.sqrt(D)/theta[2])**2/theta[1]**2)
#         K = theta[0]*np.cos(2*np.pi*D/theta[2])
#         K = theta[0]*np.cos(2*np.pi*np.sqrt(D)/theta[2])
    elif ktype == 'qp':
        K = theta[0]*np.exp(-D/(2*theta[1]**2) \
                -.5*np.sin(np.pi*np.sqrt(D)/theta[2])**2/theta[3]**2)
#         K = theta[0]*np.cos(2*np.pi*np.sqrt(D)/theta[2])*np.exp(-D/(2*theta[1]**2))
    if wn == True:
        K += np.identity(X1[:,0].size)*theta[-1]**2
    return np.matrix(K)

# # squared exponential covariance kernel
# def cf(X1, X2, theta, white_noise = False):
#     D = sp.distance.cdist(X1, X2, 'sqeuclidean')
# #     K = theta[0] * np.exp(-D/(2*(theta[1]**2))) # SE
# #     K = theta[0] * np.exp(-2*np.sin(np.pi*D/theta[2])**2/theta[1]**2) # per
#     K = theta[0]*np.exp(-(np.sin(np.pi*D/theta[2]))**2/2./theta[1]**2 \
#                   - D/2*theta[3]**2)
#     if white_noise == True:
# #         K += (np.identity(X1[:,0].size) * (theta[2]**2)) # SE
# #         K += (np.identity(X1[:,0].size) * (theta[3]**2)) # per
#         K += (np.identity(X1[:,0].size) * (theta[4]**2)) # QP
#     return np.matrix(K)

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
subsamp = 1
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
# h_init = [1., .4, .3] # SE
h_init = [1.,1.,2.,.4,.3] # quasi-periodic
# h_init = [10.,1.,2.,.3] # periodic

# h_init = [1., 6., .3] #se
# theta = [1., 3., 20., 0.3] # p
# theta = [1., 10., 20., 1., 0.3] # qp

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
#     if 0.<h[0]<10. and 0.<h[1]<10. and 0.<h[2]<1.: # SE
#     if 0.<h[0]<10. and 0.<h[1]<50. and 0.<h[2]<50. and 0.<h[3]<2.: # per
    if 0.<h[0]<100. and 0.<h[1]<100. and 0.<h[2]<100. and 0.<h[3]<10. and 0.<h[4]<2.:
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

# compute prediction for initial guess and plot
print "plotting guess"
ys, ys_err = predict(Xs, X, y,covfunc, h_init)
pl.clf()
pl.errorbar(xplot, yplot, color='k', fmt='.', capsize=0, ecolor='.7', zorder=2, alpha=.8)
pl.plot(xs, ys, color = "#339999", zorder=1, linewidth='2')
# pl.plot(xs, ys+ys_err, color = "#339999", zorder=1, linewidth='2',alpha='.5')
# pl.plot(xs, ys-ys_err, color = "#339999", zorder=1, linewidth='2', alpha='.5')
pl.xlabel("$\mathrm{Time~(days)}$")
pl.xlim(259, 272)
pl.savefig('guess')

# plot covariance matrix
pl.clf()
K = cf(X,X,h_init, ktype='p')
pl.imshow(K, interpolation = 'nearest', cmap = 'gray')
pl.savefig('K')
raw_input('enter')

# plot draws from multivariate Gaussian
ocols = ['#FF9933','#66CCCC' , '#FF33CC', '#3399FF', '#CC0066', '#99CC99', '#9933FF', '#CC0000', '#99FF33', '#66CC00']
pl.clf()
# xs -= min(xs)
pars = [1.,1.,10.,.4,.3] # quasi-periodic
# pars = [1.,1.,8.,.3] # periodic
# pars = [1., 2, .3] # SE
# K = cf(Xs, Xs, pars, ktype='se')
# K = cf(Xs, Xs, pars, ktype='p')
K = cf(Xs, Xs, pars, ktype='qp')
draw = np.random.multivariate_normal(np.zeros(len(xs)), K)
pl.plot(xs, draw, color=ocols[-1], linewidth='2')

pars = [1.,1.,3.,.4,.3] # quasi-periodic
# pars = [1.,1.,5.,.3] # periodic
# pars = [1., 1, .3] # SE
# K = cf(Xs, Xs, pars, ktype='se')
# K = cf(Xs, Xs, pars, ktype='p')
K = cf(Xs, Xs, pars, ktype='qp')
draw = np.random.multivariate_normal(np.zeros(len(xs)), K)
pl.plot(xs, draw+4, color=ocols[1], linewidth='2')

pars = [1.,1.,1.,.4,.3] # quasi-periodic
# pars = [1.,1.,3.,.3] # periodic
# pars = [1., .5, .3] # SE
# K = cf(Xs, Xs, pars, ktype='se')
# K = cf(Xs, Xs, pars, ktype='p')
K = cf(Xs, Xs, pars, ktype='qp')
draw = np.random.multivariate_normal(np.zeros(len(xs)), K)
pl.plot(xs, draw+8, color=ocols[2], linewidth='2')

pars = [1.,1.,.5,.4,.3] # quasi-periodic
# pars = [1., 1.,1.,.3] # periodic
# pars = [1., .1, .3] # SE
# K = cf(Xs, Xs, pars, ktype='se')
# K = cf(Xs, Xs, pars, ktype='p')
K = cf(Xs, Xs, pars, ktype='qp')
draw = np.random.multivariate_normal(np.zeros(len(xs)), K)
pl.plot(xs, draw+12, color=ocols[0], linewidth='2')

pl.xlabel("$\mathrm{Time~(days)}$")
pl.ylabel("$\mathrm{Flux}$")
pl.xlim(0, max(xs))
# pl.savefig('draws_se')
# pl.savefig('draws_p')
pl.savefig('draws_qp')
# pl.savefig('draws_qp2')


# # optimize the negative log likelihood wrt the covariance parameters
# print "Initial covariance parameters: ", h_init,"\n"
# print "Initial NLL: ", lnlike(h_init,X,y,covfunc),"\n"
# par = so.fmin(lnlike, h_init, (X,y,covfunc))
# print "Maximum likelihood covariance parameters: ", par,"\n"
# print "Final NLL: ", lnlike(par,X,y,covfunc),"\n"
# raw_input('enter')

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
fig_labels = ["$A$", "$l$", "$P$", "$M$", "$W$"]
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
