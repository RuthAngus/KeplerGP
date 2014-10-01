import numpy as np
import matplotlib.pyplot as plt
import george
from george.kernels import ExpSine2Kernel, ExpSquaredKernel, WhiteKernel
from scipy.optimize import minimize, fmin
import emcee
import triangle
import h5py

def predict(theta, xs, x, y, yerr, p):
    theta = np.exp(theta)
    k = theta[0] * ExpSquaredKernel(theta[1]) * ExpSine2Kernel(theta[2], p)
    gp = george.GP(k)
    gp.compute(x, np.sqrt(theta[3]+yerr**2))
    return gp.predict(y, xs)

def neglnlike(theta, x, y, yerr, p):
    theta = np.exp(theta)
    k = theta[0] * ExpSquaredKernel(theta[1]) * ExpSine2Kernel(theta[2], p)
    gp = george.GP(k)
    try:
        gp.compute(x, np.sqrt(theta[3]+yerr**2))
    except (ValueError, np.linalg.LinAlgError):
        return 10e25
    return -gp.lnlikelihood(y, quiet=True)

def lnprior(theta):
    if -20 < theta[0] < 16 and 0 < theta[1] < 16 and 0 < theta[2] < 20 \
            and -20 < theta[3] < 20:
                return 0.
    return -np.inf

def lnprob(theta, x, y, yerr, p):
    return lnprior(theta) + lnlike(theta, x, y, yerr, p)

def grid(results, x, y, yerr, periods):
    Ls = np.zeros_like(periods)
    for i, p in enumerate(periods):
        print 'minimizing'
        results = fmin(neglnlike, results, args=(x, y, yerr, p))
#         results = gp.optimize(x, y, yerr, dims=[0,1,2,3])
        Ls[i] = -neglnlike(results, x, y, yerr, p)
    return Ls

if __name__ == "__main__":

    fname = 'MOST'
#     t = 1000 # limit number of points

    # load Wasp data
    # x, y, yerr = np.genfromtxt('/Users/angusr/angusr/data/Wasp/
    #                            1SWASPJ233549.28+002643.8_J233549_300_ORFG_TAMUZ.lc',
    #                            skip_header=110).T[:,:t]

    # load MOST data
    x, y, yerr = \
            np.genfromtxt("/Users/angusr/Downloads/267HIP1164542014reduced.dat").T

    y -= np.median(y)  # subtract the median
    x -= x[0]  # zero time

    m = (x > 16.) * (x < 40.)
    x = x[m]
    y = y[m]
    yerr = yerr[m]

    # bin data per day
    bins = np.arange(0, 50, 0.5)
    inds = np.digitize(x, bins, right=False)
    b_x, b_y, b_yerr = np.zeros(max(inds)), np.zeros(max(inds)), np.zeros(max(inds))
    for i in range(len(b_x)):
        b_x[i], b_y[i] = np.mean(x[inds==i]), np.mean(y[inds==i])
        b_yerr[i] = np.sqrt(np.sum(yerr[inds==i]**2))/float(len(yerr[inds==i]))
    l = np.isfinite(b_y)
    b_x, b_y, b_yerr = b_x[l], b_y[l], b_yerr[l]

    # plot data
    plt.clf()
    plt.errorbar(x, y, yerr=yerr, fmt='k.', capsize=0, ecolor='.8')
    plt.errorbar(b_x, b_y, yerr=b_yerr, fmt='r.', capsize=0, ecolor='r')
    plt.savefig('%s_data' % fname)

    print np.var(b_y)
    print np.log(np.var(b_y))

    # initial guess
    # theta = np.log([1.**2, .5 ** 2, 100., 0.05, 16.]) # Wasp init
#     theta = np.log([1e-6, 20. ** 2, 10, 1e-7, 16]) # MOST init
    theta = np.log([1e-6, 20. ** 2, 20, 1e-7]) # MOST init

    # trial periods
    periods = 10**np.linspace(0, 2, 30)
    periods = np.linspace(1, 20, 20)
    L = grid(theta, b_x, b_y, b_yerr, periods)

    plt.clf()
    plt.plot(periods, L)
    plt.xlabel('Period (days)')
    plt.ylabel('Log likelihood')
    plt.savefig('%s_likelihood' % fname)
