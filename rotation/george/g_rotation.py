import numpy as np
import pyfits
import matplotlib.pyplot as pl
from lnlikefn import lnlike, predict
import emcee
import triangle
from load_dataGP import load
from synth import synthetic_data, simple_s_data
import scipy.optimize as so
import time

# flat priors (quasi-periodic)
def lnprior(theta):
#     if -16.<theta[0]<10. and -.1<theta[1]<3. and -6.<theta[2]<10. and -6.<theta[3]<16.\
    if -16.<theta[0]<12. and -2.<theta[1]<3. and -2.<theta[2]<2. and -2.<theta[3]<2.\
            and -1.<theta[4]<11.:
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

    # Load real data
    x, y, yerr = load("/Users/angusr/angusr/data2/Q3_public/kplr010295224-2009350155506_llc.fits")

    # shorten data
    l = 300.
    x = x[:l]
    y = y[:l]
    yerr = yerr[:l]

#     # normalise so range is 2 - no idea if this is the right thing to do...
#     yerr = 2*yerr/(max(y)-min(y))
#     y = 2*y/(max(y)-min(y))
#     y = y-np.median(y)

    # median normalise
    yerr /= np.median(y)
    y = y/np.median(y) -1

    # generate fake data
    pars = [10., 0., .5, .5, 4.]
#     y, yerr = synthetic_data(x, yerr, pars)
    y, yerr = simple_s_data(x, yerr, pars)

    # initial hyperparameters (logarithmic)
    # A, P, l2 (sin), l1 (exp)
    theta = [10., 0., .5, .5, 4.] # same as pars
#     theta = [7., 0., .5, .5, 1.] # initial
#     theta = [-1., .3, .5, .5, 1.] # better initialisation

    # plot data
    pl.clf()
    pl.errorbar(x, y, yerr=yerr, fmt='k.')
    xs = np.linspace(min(x), max(x), 500)
    pl.plot(xs, predict(xs, x, y, yerr, theta)[0], 'r-')
    pl.xlabel('time (days)')
    pl.savefig('data')
    raw_input('enter')

    print "Initial parameters = (exp)", theta
    start = time.clock()
    print "Initial lnlike = ", lnlike(theta, x, y, yerr),"\n"
    elapsed = (time.clock() - start)
    print 'time =', elapsed

    # Sample the posterior probability for m.
    nwalkers, ndim = 64, len(theta)
    p0 = [theta+1e-4*np.random.rand(ndim) for i in range(nwalkers)]
    sampler = emcee.EnsembleSampler(nwalkers, ndim, lnprob, args = (x, y, yerr))
    bi, pr = 100, 1000
#     print 'predicted time = ', (elapsed*bi+elapsed*pr)
    start = time.clock()
    print("Burn-in")
    p0, lp, state = sampler.run_mcmc(p0, bi)
    sampler.reset()
    print("Production run")
    sampler.run_mcmc(p0, pr)
    elapsed = time.clock() - start
    print 'time = ', elapsed/60., 'mins'

    print("Making triangle plots")
    fig_labels = ["$A$", "$P$", "$l_2$", "$l_1$", "$s$"]
    fig = triangle.corner(np.exp(sampler.flatchain), truths=np.exp(theta), labels=fig_labels[:len(theta)])
    fig.savefig("triangle_linear.png")
    fig = triangle.corner(sampler.flatchain, truths=theta, labels=fig_labels[:len(theta)])
    fig.savefig("triangle.png")

    print("Plotting traces")
    pl.figure()
    for i in range(ndim):
        pl.clf()
        pl.axhline(theta[i], color = "r", zorder=2)
        pl.plot(sampler.chain[:, :, i].T, 'k-', alpha=0.3, zorder=1)
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

#     # Grid over periods
#     P = np.arange(0.1, 5, 0.01)
#     P = np.log(P)
#     L = np.empty_like(P)
#
#     for i, per in enumerate(P):
#         theta[1] = per
#         L[i] = lnlike(theta, x, y, yerr)
#
#     P = np.exp(P)
#     pl.clf()
#     pl.plot(P, np.log(L), 'k-')
#     pl.xlabel('Period (days)')
#     pl.ylabel('likelihood')
#     pl.savefig('likelihood')
