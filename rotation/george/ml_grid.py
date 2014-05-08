import numpy as np
import pyfits
import matplotlib.pyplot as pl
from fixed_p_like import lnlike, predict, QP, neglnlike
import emcee
import triangle
from load_dataGP import load
from synth import synthetic_data, simple_s_data
import scipy.optimize as so
import time
from matplotlib.ticker import MaxNLocator
from scipy.optimize import fmin
from scipy.signal import periodogram

ocols = ['#FF9933','#66CCCC' , '#FF33CC', '#3399FF', '#CC0066', '#99CC99', '#9933FF', '#CC0000']
plotpar = {'axes.labelsize': 20,
           'text.fontsize': 20,
           'legend.fontsize': 15,
           'xtick.labelsize': 18,
           'ytick.labelsize': 18,
           'text.usetex': True}
pl.rcParams.update(plotpar)

# flat priors (quasi-periodic)
def lnprior(theta, b):
    if -16.<theta[0]<16. and -16.<theta[1]<16. and -16.<theta[2]<16.\
            and -16.<theta[3]<16. and theta[4]-(2*b)<theta[4]<theta[4]+(2*b):
        return 0.0
    return -np.inf

# posterior prob
def lnprob(theta, x, y, yerr, b):
    lp = lnprior(theta, b)
    if not np.isfinite(lp):
        return -np.inf
    try:
        return lp + lnlike(theta, x, y, yerr)
    except:
        print theta
        raise

# run the mcmc over a range of period values
def maxlike(theta, x, y, yerr, P, name):

    print 'initial values = ', theta, P
    print 'initial nll = ', neglnlike(theta, x, y, yerr, P)
    result = fmin(neglnlike, theta, args = (x, y, yerr, P))
    print 'final values = ', result, P
    like = neglnlike(result, x, y, yerr, P)
    print 'final likelihood = ', like

    savedata = np.empty(len(result)+2)
    savedata[:len(result)] = result
    savedata[-2] = P
    savedata[-1] = like
    np.savetxt('results/%sml_result.txt'%name, savedata)

    return -like

def MCMC(theta, x, y, yerr, b):

    # Sample the posterior probability for m.
    nwalkers, ndim = 64, len(theta)
    p0 = [theta+1e-4*np.random.rand(ndim) for i in range(nwalkers)]
    sampler = emcee.EnsembleSampler(nwalkers, ndim, lnprob, args = (x, y, yerr, b))
    bi, pr = 200, 800
    start = time.clock()
    print("Burn-in")
    p0, lp, state = sampler.run_mcmc(p0, bi)
    sampler.reset()
    print("Production run")
    sampler.run_mcmc(p0, pr)
    elapsed = time.clock() - start
    print 'time = ', elapsed/60., 'mins'

    print("Making triangle plots")
    fig_labels = ["$A$", "$l_2$", "$l_1$", "$s$", "$P$"]
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
    print 'mcmc result = ', theta

    like = lnlike(theta, x, y, yerr)
    print "Final lnlike = ", like

    # plot mcmc result
    pl.clf()
    pl.errorbar(x, y, yerr=yerr, fmt='k.')
    xs = np.arange(min(x), max(x), 0.01)
    pl.plot(xs, predict(xs, x, y, yerr, theta, theta[4])[0], 'r-')
    pl.xlabel('time (days)')
    pl.savefig('result')

if __name__ == "__main__":

    # Load target list with ACF periods
    data = np.genfromtxt('/Users/angusr/Python/george/targets.txt').T
    KIDs = data[0]
    p_init = data[1]

    for k, KID in enumerate(KIDs):

        # Load real data
        x, y, yerr = load("/Users/angusr/angusr/data2/Q3_public/kplr0%s-2009350155506_llc.fits" %int(KID))

        # shorten data
        print p_init[k]
        l = 500.
        x = x[:l]
        y = y[:l]
        yerr = yerr[:l]

        # normalise so range is 2 - no idea if this is the right thing to do...
        yerr = 2*yerr/(max(y)-min(y))
        y = 2*y/(max(y)-min(y))
        y = y-np.median(y)

    #     theta, P = [-2., -2., -1.2, 1.], 1.7 # generating fake data
        theta, P = [-2., -2., -1.2, 1.], p_init[k]

    #     # generate fake data
    #     K = QP(theta, x, yerr, P)
    #     y = np.random.multivariate_normal(np.zeros(len(x)), K)

        # plot data
        pl.clf()
        pl.errorbar(x, y, yerr=yerr, fmt='k.', capsize=0, ecolor='0.5', zorder=2)
        xs = np.linspace(min(x), max(x), 1000)
        pl.plot(xs, predict(xs, x, y, yerr, theta, P)[0], color='#339999', linestyle = '-',\
                zorder=1, linewidth='2')
        pl.xlabel('$\mathrm{Time~(days)}$')
        pl.ylabel('$\mathrm{Normalised~Flux}$')
        pl.gca().yaxis.set_major_locator(MaxNLocator(prune='lower'))
        pl.savefig('ml_data')
        raw_input('enter')

    #     # plot periodogram
    #     freq, power = periodogram(x)
    #     pl.clf()
    #     pl.plot(freq, power)
    #     pl.savefig('periodogram')
    #     raw_input('enter')

        print "Initial parameters = (exp)", theta
        start = time.clock()
        print "Initial lnlike = ", -neglnlike(theta, x, y, yerr, P),"\n"
        elapsed = (time.clock() - start)
        print 'time =', elapsed

        # Grid over periods
        step = 0.01
        Periods = np.arange(1., 3., step)
        L = np.zeros_like(Periods)

        for i, P in enumerate(Periods):
            L[i] = maxlike(theta, x, y, yerr, P, i)
            pl.clf()
            pl.plot(Periods, L, 'k-')
            pl.xlabel('Period')
            pl.ylabel('Likelihood')
            pl.savefig('ml_update')

        pl.clf()
        pl.plot(Periods, L, 'k-')
        pl.xlabel('Period')
        pl.ylabel('Likelihood')
        pl.savefig('ml_likelihood')

        np.savetxt('ml_results.txt', np.transpose((Periods, L)))

        mlp = Periods[L == max(L)]
        print 'max liklihood period = ', mlp

        # running MCMC over maxlikelihood period
    #     mlp = 1.41
    #     m = np.empty(len(theta)+1)
    #     m[:len(theta)] = theta
    #     m[-1] = mlp
    #     MCMC(m, x, y, yerr, b)
