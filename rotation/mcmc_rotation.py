import numpy as np
import pyfits
# from savefig import monkey_patch
# monkey_patch()
import matplotlib.pyplot as pl
from lnlikefn import lnlike, predict
import emcee
import triangle

# flat priors (quasi-periodic)
def lnprior(theta):
#     if -16.<theta[0]<10. and -1.<theta[1]<1.7 and -6.<theta[2]<10. and -1.6<theta[3]<10.:
    if -16.<theta[0]<10. and -1.<theta[1]<1.7 and -6.<theta[2]<10. and -6.<theta[3]<16.:
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

    # Load data
#     hdulist = pyfits.open("/Users/angusr/angusr/data2/Q3_public/kplr003223000-2009350155506_llc.fits")
    hdulist = pyfits.open("/Users/angusr/angusr/data2/Q3_public/kplr010295224-2009350155506_llc.fits")
    tbdata = hdulist[1].data
    x = tbdata["TIME"]
    y = tbdata["PDCSAP_FLUX"]
    yerr = tbdata["PDCSAP_FLUX_ERR"]
    q = tbdata["SAP_QUALITY"]

    # remove nans
    n = np.isfinite(x)*np.isfinite(y)*np.isfinite(yerr)*(q==0)
    l = 300.
    x = x[n][:l]
    y = y[n][:l]
    yerr = yerr[n][:l]
    mu = np.median(y)
    y = y/mu - 1.
    yerr /= mu
    yerr *= 10.

    # subsample
    subsamp = 1
    x = x[0:-1:subsamp]
    y = y[0:-1:subsamp]
    yerr = yerr[0:-1:subsamp]

    # initial hyperparameters (logarithmic)
    # A, P, l2 (sin), l1 (exp)
#     theta = [-14.2, -1.85, 2.5, -1.] # better initialisation - Ruth 3223000
#     theta = [-14.2, 0.69, 2.5, -1.] # starting at correct period 3223000
#     theta = [-15., 0.69, .5, -1.] # restricting l2 3223000
#     theta = [-14., -0.9, .5, -1.] # 10295224
    theta = [-14., .4, .5, -1.] # 10295224
#     theta = [-14., .4, .5, 5.] # 10295224 copying parameters from p_grid

    pl.clf()
    pl.errorbar(x, y, yerr=yerr, fmt='k.')
    xs = np.linspace(min(x), max(x), 500)
    pl.plot(xs, predict(xs, x, y, yerr, theta)[0], 'r-')
    pl.xlabel('time (days)')
    pl.savefig('data')

    print "Initial parameters = (exp)", theta
    print "Initial lnlike = ", lnlike(theta, x, y, yerr),"\n"

    # Sample the posterior probability for m.
    nwalkers, ndim = 64, len(theta)
    p0 = [theta+1e-4*np.random.rand(ndim) for i in range(nwalkers)]
    sampler = emcee.EnsembleSampler(nwalkers, ndim, lnprob, args = (x, y, yerr))
    print("Burn-in")
    p0, lp, state = sampler.run_mcmc(p0, 200)
    sampler.reset()
    print("Production run")
    sampler.run_mcmc(p0, 1500)

    print("Making triangle plots")
    fig_labels = ["$A$", "$P$", "$l_2$", "$l_1$"]
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

    # Grid over periods
    P = np.arange(0.1, 5, 0.01)
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
