import numpy as np
import pyfits
import matplotlib.pyplot as pl
from fixed_p_like import lnlike, predict
import emcee
import triangle
from load_dataGP import load
from synth import synthetic_data, simple_s_data
import scipy.optimize as so
import time
from matplotlib.ticker import MaxNLocator

ocols = ['#FF9933','#66CCCC' , '#FF33CC', '#3399FF', '#CC0066', '#99CC99', '#9933FF', '#CC0000']
plotpar = {'axes.labelsize': 20,
           'text.fontsize': 20,
           'legend.fontsize': 15,
           'xtick.labelsize': 18,
           'ytick.labelsize': 18,
           'text.usetex': True}
pl.rcParams.update(plotpar)

# flat priors (quasi-periodic)
def lnprior(theta):
    if -10.<theta[0]<10. and -10.<theta[1]<10. and -10.<theta[2]<10.\
            and -10.<theta[3]<10.:
        return 0.0
    return -np.inf

# posterior prob
def lnprob(theta, x, y, yerr, P):
    lp = lnprior(theta)
    if not np.isfinite(lp):
        return -np.inf
    try:
        return lp + lnlike(theta, x, y, yerr, P)
    except:
        print theta
        raise

# run the mcmc over a range of period values
def MCMC(theta, x, y, yerr, P):

    # Sample the posterior probability for m.
    nwalkers, ndim = 64, len(theta)
    p0 = [theta+1e-4*np.random.rand(ndim) for i in range(nwalkers)]
    sampler = emcee.EnsembleSampler(nwalkers, ndim, lnprob, args = (x, y, yerr, P))
    bi, pr = 100, 500
    start = time.clock()
    print("Burn-in")
    p0, lp, state = sampler.run_mcmc(p0, bi)
    sampler.reset()
    print("Production run")
    sampler.run_mcmc(p0, pr)
    elapsed = time.clock() - start
    print 'time = ', elapsed/60., 'mins'

    print("Making triangle plots")
    fig_labels = ["$A$", "$l_2$", "$l_1$", "$s$"]
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

    print "Final lnlike = ", lnlike(theta, x, y, yerr, P)

    # plot mcmc result
    pl.clf()
    pl.errorbar(x, y, yerr=yerr, fmt='k.')
    xs = np.arange(min(x), max(x), 0.01)
    pl.plot(xs, predict(xs, x, y, yerr, theta, P)[0], 'r-')
    pl.xlabel('time (days)')
    print 'P=', np.log10(P)
    raw_input('enter')
    pl.savefig('%sresult'%np.log10(P))

    return lnlike(theta, x, y, yerr, P)

if __name__ == "__main__":
    # Load real data
    x, y, yerr = load("/Users/angusr/angusr/data2/Q3_public/kplr010295224-2009350155506_llc.fits")

    # shorten data
    l = 550.
    x = x[:l]
    y = y[:l]
    yerr = yerr[:l]

    # normalise so range is 2 - no idea if this is the right thing to do...
    yerr = 2*yerr/(max(y)-min(y))
    y = 2*y/(max(y)-min(y))
    y = y-np.median(y)

#     theta, P = [0., .2, .2, 1.], 1.7 # initial
    theta, P = [-2., -2., -1.2, 6.], 1.7 # better initialisation

    # plot data
    pl.clf()
    pl.errorbar(x, y, yerr=yerr, fmt='k.', capsize=0, ecolor='0.5', zorder=2)
    xs = np.linspace(min(x), max(x), 1000)
    pl.plot(xs, predict(xs, x, y, yerr, theta, P)[0], color='#339999', linestyle = '-',\
            zorder=1, linewidth='2')
    pl.xlabel('$\mathrm{Time~(days)}$')
    pl.ylabel('$\mathrm{Normalised~Flux}$')
    pl.gca().yaxis.set_major_locator(MaxNLocator(prune='lower'))
    pl.savefig('data')
    raw_input('enter')

    print "Initial parameters = (exp)", theta
    start = time.clock()
    print "Initial lnlike = ", lnlike(theta, x, y, yerr, P),"\n"
    elapsed = (time.clock() - start)
    print 'time =', elapsed

    # Grid over periods
    Periods = np.arange(0.1, 5, 0.01)
    L = np.empty_like(Periods)

    for i, P in enumerate(Periods):
        L[i] = MCMC(theta, x, y, yerr, P)
        raw_input('enter')

    pl.clf()
    pl.plot(Periods, L, 'k-')
    pl.xlabel('Period')
    pl.ylabel('Likelihood')
    pl.savefig('likelihood')
