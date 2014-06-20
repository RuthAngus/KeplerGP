import numpy as np
import matplotlib.pyplot as pl
from fixed_p_like import binary_lnlike, binary_neglnlike, binary_predict
from fixed_p_like import lnlike, neglnlike, predict
import pyfits
from injection_tests import subs
import emcee
import triangle

def lnprior(theta, bm, bp):
    b = 16.
    if -16.<theta[0]<16. and -16.<theta[1]<16. and -16.<theta[2]<16.\
            and -16.<theta[3]<16. and bm<theta[4]<bp and -16.<theta[5]<16.\
            and -16.<theta[6]<16. and -16.<theta[7]<16. and bm<theta[8]<bp \
            and theta[4]>theta[8]:
        return 0.0
    return -np.inf

def lnprob(theta, x, y, yerr, bm, bp):
    lp = lnprior(theta, bm, bp)
    if not np.isfinite(lp):
        return -np.inf
    try:
        return lp + binary_lnlike(theta, x, y, yerr)
    except:
        print theta
        raise

# load data
hdulist = pyfits.open("/Users/angusr/.kplr/data/lightcurves/003454793/kplr003454793-2009350155506_llc.fits")
tbdata = hdulist[1].data
x = tbdata["TIME"]
y = tbdata["PDCSAP_FLUX"]
yerr = tbdata["PDCSAP_FLUX_ERR"]
q = tbdata["SAP_QUALITY"]
# remove nans and bad flags
n = np.isfinite(x)*np.isfinite(y)*np.isfinite(yerr)*(q==0)
x = x[n]
y = y[n]
yerr = yerr[n]

# normalise so range is 2 - no idea if this is the right thing to do...
yerr = 2*yerr/(max(y)-min(y))
y = 2*y/(max(y)-min(y))
y = y-np.median(y)

# subsample and truncate
x_sub, y_sub, yerr_sub = subs(x, y, yerr, 1., 500)

# A_1, l2_1, l1_1, s, P_1, A_2, l2_2, l1_2, P_2
# theta = [-2., -2., -1.2, 1., np.log(.66), -2., -2., -1.2, -1.2, np.log(.97)]

# better initialisation
theta = [-7., -3., -7., 1., np.log(.66), -3.5, -2., -2., -1.6, np.log(.97)]

# plot data and prediction
pl.clf()
pl.errorbar(x_sub, y_sub, yerr=yerr_sub, fmt='k.', capsize=0)
xs = np.linspace(min(x_sub), max(x_sub), 100)
pl.plot(xs, binary_predict(xs, x_sub, y_sub, yerr_sub, theta)[0], color='.7')
pl.savefig('binary_data')

# Compute initial likelihood
print 'initial lnlike = ', binary_lnlike(theta, x, y, yerr)

# bm, bp = minimum and maximum periods
bm, bp = np.log(.2), np.log(2)

# Sample the posterior probability for m.
nwalkers, ndim = 64, len(theta)
p0 = [theta+1e-4*np.random.rand(ndim) for i in range(nwalkers)]
sampler = emcee.EnsembleSampler(nwalkers, ndim, lnprob, args = (x, y, yerr, bm, bp))

# print("Burn-in")
# p0, lp, state = sampler.run_mcmc(p0, 200)
# sampler.reset()

nstep = 2000
nruns = 100.

print("Production run")
fname = 'binary2'
for j in range(int(nstep/nruns)):

    print 'run', j
    p0, lp, state = sampler.run_mcmc(p0, nruns)

    print("Plotting traces")
    pl.figure()
    for i in range(ndim):
        pl.clf()
        pl.axhline(theta[i], color = "r")
        pl.plot(sampler.chain[:, :, i].T, 'k-', alpha=0.3)
        pl.savefig("%s%s.png" %(i, fname))

    flat = sampler.chain[:, 50:, :].reshape((-1, ndim))
    mcmc_result = map(lambda v: (v[1], v[2]-v[1], v[1]-v[0]),
                      zip(*np.percentile(flat, [16, 50, 84], axis=0)))
    print mcmc_result
    np.savetxt("mcmc_result%s.txt"%fname, mcmc_result)
    mres = np.array(mcmc_result)[:, 0]
    print 'mcmc_result = ', mres

    print("Making triangle plot")
    fig_labels = ["$A_1$", "$l2_1$", "$l1_1$", "$s$", "$P_1$", "$A_2$", "$l2_2$", \
            "$l1_2$", "$P_2$", "$a$"]
    fig = triangle.corner(sampler.flatchain, truths=theta, labels=fig_labels[:len(theta)])
    fig.savefig("triangle_%s.png" %fname)

# Flatten chain
samples = sampler.chain[:, 50:, :].reshape((-1, ndim))

# Find values
mcmc_result = map(lambda v: (v[1], v[2]-v[1], v[1]-v[0]),
                  zip(*np.percentile(samples, [16, 50, 84], axis=0)))

theta = np.array(mcmc_result)[:, 0]
print 'mcmc result = ', theta

like = lnlike(theta, x, y, yerr)
print "Final lnlike = ", like

# plot mcmc result
pl.clf()
pl.errorbar(x, y, yerr=yerr, fmt='k.')
xs = np.arange(min(x), max(x), 0.01)
pl.plot(xs, predict(xs, x, y, yerr, theta, theta[4])[0], 'r-')
pl.xlabel('time (days)')
pl.savefig('result%s'%fname)
