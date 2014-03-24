import numpy as np
import pyfits
import matplotlib.pyplot as pl
from lnlikefn import lnlike
import emcee
import triangle

# Load data
hdulist = pyfits.open("/Users/angusr/angusr/data2/Q3_public/kplr003223000-2009350155506_llc.fits")
tbdata = hdulist[1].data
x = tbdata["TIME"]
y = tbdata["PDCSAP_FLUX"]
yerr = tbdata["PDCSAP_FLUX_ERR"]
q = tbdata["SAP_QUALITY"]

# remove nans
n = np.isfinite(x)*np.isfinite(y)*np.isfinite(yerr)*(q==0)
l = 500.
x = x[n][:l]
y = y[n][:l]
yerr = yerr[n][:l]
mu = np.median(y)
y = y/mu - 1.
yerr /= mu

pl.clf()
pl.errorbar(x, y, yerr=yerr, fmt='k.')
pl.xlabel('time (days)')
pl.savefig('data')

# uniform priors (quasi-periodic)
def lnprior(h):
    if 0.<h[0]<100. and 0.<h[1]<100. and 0.<h[2]<100. and 0.<h[3]<10.:
        return 0.0
    return -np.inf

# posterior prob
def lnprob(h, x, y, yerr):
    lp = lnprior(h)
    if not np.isfinite(lp):
        return -np.inf
    return lp + lnlike(h, x, y, yerr)

theta = [1e-8, 1., 10., 2.]
# theta = [1e-8, 10., 2.]

print "Initial parameters = ", theta
print "Initial lnlike = ", lnlike(theta, x, y, yerr),"\n"

# Sample the posterior probability for m.
nwalkers, ndim = 32, len(theta)
p0 = [theta+1e-4*np.random.rand(ndim) for i in range(nwalkers)]
sampler = emcee.EnsembleSampler(nwalkers, ndim, lnprob, args = (x, y, yerr))
print("Burn-in")
p0, lp, state = sampler.run_mcmc(p0, 100)
sampler.reset()
print("Production run")
sampler.run_mcmc(p0, 1000)

print("Making triangle plot")
fig_labels = ["$A$", "$l_1$", "$l_2$", "$P$"]
fig = triangle.corner(sampler.flatchain, truths=theta, labels=fig_labels[:len(theta)])
fig.savefig("triangle.png")

print("Plotting traces")
pl.figure()
for i in range(ndim):
    pl.clf()
    pl.axhline(theta[i], color = "r")
    pl.plot(sampler.chain[:, :, i].T, 'k-', alpha=0.3)
    pl.savefig("{0}.png".format(i))

# Flatten chain
samples = sampler.chain[:, 50:, :].reshape((-1, ndim))

# Find values
mcmc_result = map(lambda v: (v[1], v[2]-v[1], v[1]-v[0]),
                  zip(*np.percentile(samples, [16, 50, 84], axis=0)))

h = np.array(mcmc_result)[:, 0]
print 'mcmc result', h

print "Final lnlike = ", lnlike(h, x, y, yerr)

# P = np.arange(0.1, 5, 0.1)
# L = np.empty_like(P)

# for i, per in enumerate(P):
#     theta[1] = per
#     L[i] = lnlike(theta, x, y, yerr)

# pl.clf()
# pl.plot(P, L, 'k-')
# pl.xlabel('Period (days)')
# pl.ylabel('likelihood')
# pl.savefig('likelihood')
