import numpy as np
import matplotlib.pyplot as plt
import george
from george.kernels import ExpSine2Kernel, ExpSquaredKernel, WhiteKernel
from scipy.optimize import minimize, fmin
import emcee
import triangle
import h5py
from rotation import predict, neglnlike

plotpar = {'axes.labelsize': 15,
           'text.fontsize': 20,
           'legend.fontsize': 15,
           'xtick.labelsize': 18,
           'ytick.labelsize': 18,
           'text.usetex': True}
plt.rcParams.update(plotpar)

ocols = ['#FF9933','#66CCCC' , '#FF33CC', '#3399FF', '#CC0066',
'#99CC99', '#9933FF', '#CC0000', '#99CC00']

# def grid(results, x, y, yerr, periods):
def grid(theta, x, y, yerr, periods, lhf, fname):
    L = np.zeros_like(periods)
    results = np.zeros((len(theta), len(L)))
    for i, p in enumerate(periods):
        print 'minimizing'
        result = fmin(lhf, theta, args=(x, y, yerr, p))
        L[i] = -lhf(result, x, y, yerr, p)
        results[:, i] = result

    l = L==max(L)
    new_theta = results[:, l].T[0]
    period = periods[l][0]
    print 'max likelihood = ', L[l][0], 'period = ', period
    print 'best params: ', new_theta

    plt.clf()
    plt.subplot(2,1,1)
    plt.plot(periods, L, color=ocols[1])
    plt.xlabel('$\mathrm{Period~(days)}$')
    plt.ylabel('$\mathrm{Log~likelihood}$')
    plt.savefig('%s_likelihood' % fname)

    plt.subplot(2,1,2)
    plt.errorbar(b_x, b_y, yerr=b_yerr, fmt='k.', capsize=0, ecolor='.8')
    xs = np.linspace(min(x), max(x), 100)
    mu, cov = predict(new_theta, xs, x, y, yerr, period)
    plt.plot(xs, mu, color=ocols[0], label="$\mathrm{Period}=%s$" % period)
    plt.xlabel('$\mathrm{Time~(days)}$')
    plt.ylabel('$\mathrm{RV~(ms}_{-1}\mathrm{)}$')
    plt.subplots_adjust(hspace=.2)
    plt.legend()
    plt.savefig('%s_result' % fname)

    return L, results

def bin_data(x, y, yerr, minb, maxb, increment):
    bins = np.arange(minb, maxb, increment)
    inds = np.digitize(x, bins, right=False)
    b_x, b_y, b_yerr = np.zeros(max(inds)), np.zeros(max(inds)), np.zeros(max(inds))
    for i in range(len(b_x)):
        b_x[i], b_y[i] = np.mean(x[inds==i]), np.mean(y[inds==i])
        b_yerr[i] = np.sqrt(np.sum(yerr[inds==i]**2))/float(len(yerr[inds==i]))
    l = np.isfinite(b_y)
    b_x, b_y, b_yerr = b_x[l], b_y[l], b_yerr[l]
    return b_x, b_y, b_yerr

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

    # select least noisy region
    m = (x > 16.) * (x < 40.)
    x = x[m]
    y = y[m]
    yerr = yerr[m]

    # bin data per day
    b_x, b_y, b_yerr = bin_data(x, y, yerr, 0, 50, .5)

    # initial guess
    # theta = np.log([1.**2, .5 ** 2, 100., 0.05, 16.]) # Wasp init
    theta = np.log([1e-6, 20. ** 2, 20, 1e-7]) # MOST init

    # trial periods
#     periods = 10**np.linspace(0, 2, 30)  # round 1
#     periods = np.linspace(1, 20, 20)  # round 2
#     periods = np.linspace(10, 20, 20)  # round 3
    periods = np.linspace(12, 17, 20)  # round 4

    L, results = grid(theta, b_x, b_y, b_yerr, periods, neglnlike, fname)  # MOST_result.png
