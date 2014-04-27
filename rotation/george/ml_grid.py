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
    if -16.<theta[0]<16. and -16.<theta[1]<16. and -16.<theta[2]<16.\
            and -16.<theta[3]<16.:
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
def maxlike(theta, x, y, yerr, P, name):

    print 'initial values = ', theta, P
    print 'initial nll = ', neglnlike(theta, x, y, yerr, P)
    result = fmin(neglnlike, theta, args = (x, y, yerr, P))
    print 'final values = ', result, P
    like = neglnlike(result, x, y, yerr, P)
    print 'final nll = ', like

    # plot ml result
    pl.clf()
    pl.errorbar(x, y, yerr=yerr, fmt='k.')
    xs = np.arange(min(x), max(x), 0.01)
    pl.plot(xs, predict(xs, x, y, yerr, result, P)[0], 'r-')
    pl.xlabel('time (days)')
    pl.savefig('%sml_result'%name)

    savedata = np.empty(len(result)+1)
    savedata[:len(result)] = result
    savedata[-1] = like
    np.savetxt('%sml_result.txt'%name, savedata)

    return like

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
#     theta, P = [-2., -2., -1.2, 6.], 1.7 # better initialisation
    theta, P = [-2., -2., -1.2, 1.], 1.7 # generating fake data

    # generate fake data
    K = QP(theta, x, yerr, P)
    y = np.random.multivariate_normal(np.zeros(len(x)), K)

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

    print "Initial parameters = (exp)", theta
    start = time.clock()
    print "Initial lnlike = ", lnlike(theta, x, y, yerr, P),"\n"
    elapsed = (time.clock() - start)
    print 'time =', elapsed

    # Grid over periods
    Periods = np.arange(0.2, 5, 0.1)
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
