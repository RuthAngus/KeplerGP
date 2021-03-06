import numpy as np
import matplotlib.pyplot as plt
import george
from george.kernels import ExpSine2Kernel, ExpSquaredKernel, WhiteKernel
from scipy.optimize import minimize, fmin
import emcee
import triangle
import h5py

def predict(theta, xs, x, y, yerr):
#     theta = np.exp(theta)
    k = theta[0] * ExpSquaredKernel(theta[1]) * ExpSine2Kernel(theta[2], theta[4])
    gp = george.GP(k)
    gp.compute(x, np.sqrt(theta[3]+yerr**2))
    return gp.predict(y, xs)

def lnlike(theta, x, y, yerr):
#     theta = np.exp(theta)
    k = theta[0] * ExpSquaredKernel(theta[1]) * ExpSine2Kernel(theta[2], theta[4])
    gp = george.GP(k)
    try:
        gp.compute(x, np.sqrt(theta[3]+yerr**2))
    except (ValueError, np.linalg.LinAlgError):
        return 10e25
    return gp.lnlikelihood(y, quiet=True)

# theta = [-12.619, 6.153, -0.65858, -15.471, 14.1] # best init
def lnprior(theta):
    if -20 < theta[0] < 16 and 0 < theta[1] < 10 and -10 < theta[2] < 10 \
            and -20 < theta[3] < 20 and 10 < theta[4] < 18:
                return 0.
    return -np.inf

def lnprob(theta, x, y, yerr):
    return lnprior(theta) + lnlike(theta, x, y, yerr)

def MCMC(theta, x, y, yerr, fname, burn_in, nsteps, nruns):

    # calculate initial likelihood and plot initial hparams
    xs = np.linspace(min(x), max(x), 1000)
    k = theta[0] * ExpSquaredKernel(theta[1]) * ExpSine2Kernel(theta[2], theta[4])
    k += WhiteKernel(theta[3])
    gp = george.GP(k)
    print 'initial lnlike = ', lnlike(theta, x, y, yerr)
    mu, cov = predict(theta, xs, x, y, yerr)
    plt.clf()
    plt.errorbar(x, y, yerr=yerr, fmt='k.', capsize=0)
    plt.plot(xs, mu, 'r')
    std = np.sqrt(np.diag(cov))
#     plt.fill_between(mu-std, mu+std, color='r', alpha='.5')
    plt.savefig('%s_init' % fname)

    # setup sampler
    nwalkers, ndim = 32, len(theta)
    p0 = [theta+1e-4*np.random.rand(ndim) for i in range(nwalkers)]
    args = [x, y, yerr]
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
    fig_labels = ["$A$", "$l1$", "$l2$", "$wn$", "$P$"]
    fig = triangle.corner(samples, truths=mres, labels=fig_labels)
    fig.savefig("triangle_%s.png" % fname)

    # plot result
    mu, cov = predict(mres, xs, x, y, yerr)
    plt.clf()
    plt.errorbar(x, y, yerr=yerr, fmt='k.', capsize=0)
    plt.plot(xs, mu, 'r')
    plt.savefig('%s_final' % fname)

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
        b_x[i] = np.mean(x[inds==i])
        b_y[i] = np.mean(y[inds==i])
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
#     theta = np.log([1e-6, 20. ** 2, 20, 1e-7, 16]) # MOST init
    theta = [-12.619, 6.153, -0.65858, -15.471, 14.1] # best init

    # Run MCMC
    burn_in, nsteps, nruns = 1000, 2000, 10
    MCMC(theta, b_x, b_y, b_yerr, fname, burn_in, nsteps, nruns)
