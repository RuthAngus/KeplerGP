import numpy as np
import pyfits
# from savefig import monkey_patch
# monkey_patch()
import matplotlib.pyplot as pl
from lnlikefn import lnlike, predict
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

# subsample
subsamp = 5
x = x[0:-1:subsamp]
y = y[0:-1:subsamp]
yerr = yerr[0:-1:subsamp]

# flat priors (quasi-periodic)
def lnprior(theta):
    if -16.<theta[0]<10. and -6.<theta[1]<10. and -6.<theta[2]<10. and -6.<theta[3]<10.:
        return 0.0
    return -np.inf

# posterior prob
def lnprob(theta, x, y, yerr):
    lp = lnprior(theta)
    if not np.isfinite(lp):
        return -np.inf
    try:
        return lp + lnlike(theta, x, y, yerr)
    except:
        print theta
        raise

if __name__ == "__main__":

    # initial hyperparameters
    # A, P, l2 (sin), l1 (exp)

    # Linear theta
    # theta = [1e-8, 1., 10., 2.] # initial try
    # theta = [1e-8, 10., 2.] # fixed period
    # theta = np.log(theta)

    # exponential theta
    # theta = [-14., np.log(2.5), -3.0, np.log(100.0)] # better initialisation - Dan
    theta = [-14., -2., 3., -1] # better initialisation - Ruth

    pl.clf()
    pl.errorbar(x, y, yerr=yerr, fmt='k.')
    xs = np.linspace(min(x), max(x), 500)
    pl.plot(xs, predict(xs, x, y, yerr, theta)[0], 'r-')
    pl.xlabel('time (days)')
    pl.savefig('data')

    print "Initial parameters = (exp)", theta
    print "Initial lnlike = ", lnlike(theta, x, y, yerr),"\n"
    raw_input('enter')

    # Sample the posterior probability for m.
    nwalkers, ndim = 64, len(theta)
    p0 = [theta+1e-4*np.random.rand(ndim) for i in range(nwalkers)]
    sampler = emcee.EnsembleSampler(nwalkers, ndim, lnprob, args = (x, y, yerr))
    print("Burn-in")
    p0, lp, state = sampler.run_mcmc(p0, 100)
    sampler.reset()
    print("Production run")
    sampler.run_mcmc(p0, 2000)

    print("Making triangle plot")
    fig_labels = ["$A$", "$P$", "$l_1$", "$l_2$"]
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

    theta = np.array(mcmc_result)[:, 0]
    print 'mcmc result (exp) = ', np.exp(theta)
    print 'mcmc result (lin) = ', theta

    print "Final lnlike = ", lnlike(theta, x, y, yerr)

    # plot mcmc result
    pl.clf()
    pl.errorbar(x, y, yerr=yerr, fmt='k.')
    xs = np.arange(min(x), max(x), 0.01)
    pl.plot(xs, predict(xs, x, y, yerr, theta)[0], 'r-')
    pl.xlabel('time (days)')
    pl.savefig('result')

    # Grid over periods
    P = np.arange(0.1, 5, 0.1)
    P = np.log(P)
    L = np.empty_like(P)

    for i, per in enumerate(P):
        theta[1] = per
        L[i] = lnlike(theta, x, y, yerr)

    P = np.exp(P)
    pl.clf()
    pl.plot(P, L, 'k-')
    pl.xlabel('Period (days)')
    pl.ylabel('likelihood')
    pl.savefig('likelihood')