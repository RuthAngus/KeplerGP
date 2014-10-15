import numpy as np
import matplotlib.pyplot as plt
import george
from george.kernels import ExpSine2Kernel, ExpSquaredKernel, WhiteKernel
from scipy.optimize import minimize, fmin
import emcee
import triangle
import h5py
from rotation import multilnlike_emcee_wasp, predict
from GPgrid import bin_data
from colors import plot_colors
ocols = plot_colors()

def lnprior(theta):
    if -20 < theta[0] < 16 and 0 < theta[1] < 10 and -20 < theta[2] < 20 \
    and -20 < theta[3] < 20 and -20 < theta[4] < 20 and -20 < theta[5] < 20 \
    and -20 < theta[6] < 20 and -20 < theta[7] < 20 \
    and np.log(10) < theta[8] < np.log(15):
        return 0.
    return -np.inf

def lnprob(theta, x1, x2, x3, x4, x5, y1, y2, y3, y4, \
           y5, yerr1, yerr2, yerr3, yerr4, yerr5):
    return lnprior(theta) + multilnlike_emcee_wasp(theta, x1, x2, x3, x4, x5, y1, y2, \
                                        y3, y4, y5, yerr1, yerr2, yerr3, yerr4, yerr5)

def MCMC(theta, x1, x2, x3, x4, x5, y1, y2, y3, y4, y5, yerr1, yerr2, yerr3, \
         yerr4, yerr5, fname, burn_in, nsteps, nruns):

    # setup sampler
    nwalkers, ndim = 32, len(theta)
    p0 = [theta+1e-4*np.random.rand(ndim) for i in range(nwalkers)]
    args = [x1, x2, x3, x4, x5, y1, y2, y3, y4, y5, yerr1, yerr2, yerr3, yerr4, yerr5]
    sampler = emcee.EnsembleSampler(nwalkers, ndim, lnprob, args=args)

    print("Burning in...")
    p0, lp, state = sampler.run_mcmc(p0, burn_in)
    sampler.reset()

    for i in range(nruns):

        print 'Running... ', i
        p0, lp, state = sampler.run_mcmc(p0, nsteps)

        # results
        samples = sampler.chain[:, 50:, :].reshape((-1, ndim))
        mcmc_result = map(lambda v: (v[1], v[2]-v[1], v[1]-v[0]),
                          zip(*np.percentile(samples, [16, 50, 84], axis=0)))
        mres = np.array(mcmc_result)[:, 0]
        print 'mcmc_result = ', np.exp(mres)
        np.savetxt("parameters_%s.txt" % fname, np.array(mcmc_result))

        print "saving samples"
        f = h5py.File("samples%s" % fname, "w")
        data = f.create_dataset("samples", np.shape(sampler.chain))
        data[:,:] = np.array(sampler.chain)
        f.close()

    # make triangle plot
    fig_labels = ["$A$", "$l1$", "$l2$", "$wn1$", "$wn2$", "$wn3$", "$wn4$", "$wn5$", "$P$"]
    fig = triangle.corner(samples, truths=mres, labels=fig_labels)
    fig.savefig("triangle_%s.png" % fname)

    # plot results
    new_theta, period = np.exp(mres), np.exp(mres[-1])
    plt.clf()
    plt.subplot(4,1,1)
    plt.errorbar(x1, y1, yerr=yerr1, fmt='k.', capsize=0, ecolor='.8')
    xs = np.linspace(min(x1), max(x1), 100)
    pars = np.array([new_theta[0], new_theta[1], new_theta[2], new_theta[3], period])
    mu, cov = predict(pars, xs, x1, y1, yerr1, period)
    plt.plot(xs, mu, color=ocols.lightblue)

    plt.subplot(4,1,2)
    plt.errorbar(x2, y2, yerr=yerr2, fmt='k.', capsize=0, ecolor='.8')
    xs = np.linspace(min(x2), max(x2), 100)
    pars = np.array([new_theta[0], new_theta[1], new_theta[2], new_theta[4], period])
    mu, cov = predict(pars, xs, x2, y2, yerr2, period)
    plt.plot(xs, mu, color=ocols.orange)

    plt.subplot(4,1,3)
    plt.errorbar(x3, y3, yerr=yerr3, fmt='k.', capsize=0, ecolor='.8')
    xs = np.linspace(min(x3), max(x3), 100)
    pars = np.array([new_theta[0], new_theta[1], new_theta[2], new_theta[5], period])
    mu, cov = predict(pars, xs, x3, y3, yerr3, period)
    plt.plot(xs, mu, color=ocols.pink)

    plt.subplot(4,1,4)
    plt.errorbar(x4, y4, yerr=yerr4, fmt='k.', capsize=0, ecolor='.8')
    xs = np.linspace(min(x4), max(x4), 100)
    pars = np.array([new_theta[0], new_theta[1], new_theta[2], new_theta[6], period])
    mu, cov = predict(pars, xs, x4, y4, yerr4, period)
    plt.plot(xs, mu, color=ocols.green)
    plt.savefig('all_data_results_emcee_wasp')

if __name__ == "__main__":

    fname = 'wasp'

    # load Wasp data
    xw, yw, yerrw = np.genfromtxt('/Users/angusr/angusr/data/Wasp/1SWASPJ233549.28+002643.8_J233549_300_ORFG_TAMUZ.lc', skip_header=110).T

    # load MOST data
    xM, yM, yerrM = \
            np.genfromtxt("/Users/angusr/Downloads/267HIP1164542014reduced.dat").T

    # load K2 data
    xK2, yK2 = np.genfromtxt('/Users/angusr/angusr/data/Wasp/hip_mod.csv').T
    yerrK2 = np.ones_like(yK2)*.0001

    xK2 = xK2 + 2454833 - 2450000
    yK2 -= np.median(yK2)

    l = xM < 5342
    x1, y1, yerr1 = xM[l], yM[l], yerrM[l]
    l = (5342<xM) * (xM<5366)
    x2, y2, yerr2 = xM[l], yM[l], yerrM[l]
    l = 5366 < xM
    x3, y3, yerr3 = xM[l], yM[l], yerrM[l]
    x4, y4, yerr4 = xK2, yK2, yerrK2
    x5, y5, yerr5 = xw, yw, yerrw

    # bin data?
    bin_dat = True
    intrvl = .1
    if bin_dat:
        x1, y1, yerr1 = bin_data(x1, y1, yerr1, min(x1), max(x1), intrvl)
        x2, y2, yerr2 = bin_data(x2, y2, yerr2, min(x2), max(x2), intrvl)
        x3, y3, yerr3 = bin_data(x3, y3, yerr3, min(x3), max(x3), intrvl)
        x4, y4, yerr4 = bin_data(x4, y4, yerr4, min(x4), max(x4), intrvl)
    x5, y5, yerr5 = bin_data(x5, y5, yerr5, min(x5), max(x5), 10)
    print len(x5)

    # plot data
    plt.clf()
    plt.subplot(5,1,1)
    plt.errorbar(x1, y1, yerr=yerr1, fmt='k.', capsize=0, ecolor='.8')
    plt.subplot(5,1,2)
    plt.errorbar(x2, y2, yerr=yerr2, fmt='k.', capsize=0, ecolor='.8')
    plt.subplot(5,1,3)
    plt.errorbar(x3, y3, yerr=yerr3, fmt='k.', capsize=0, ecolor='.8')
    plt.subplot(5,1,4)
    plt.errorbar(x4, y4, yerr=yerr4, fmt='k.', capsize=0, ecolor='.8')
    plt.subplot(5,1,5)
    plt.errorbar(x5, y5, yerr=yerr5, fmt='k.', capsize=0, ecolor='.8')
    plt.savefig('inc_wasp')

    # initial guess
    theta = np.log([1e-6, 20. ** 2, 20, 1e-7, 1e-7, \
                   1e-7, 1e-7, 1e-7, 13.])

    # Run MCMC
    burn_in, nsteps, nruns = 1000, 5000, 5
    MCMC(theta, x1, x2, x3, x4, x5, y1, y2, y3, y4, y5,
         yerr1, yerr2, yerr3, yerr4, yerr5, fname, burn_in,
         nsteps, nruns)
