import numpy as np
import matplotlib.pyplot as plt
from colors import plot_colors
from rotation import before_and_after
from GPgrid import grid, bin_data, neglnlike
import george
from george.kernels import ExpSine2Kernel, ExpSquaredKernel, WhiteKernel

plotpar = {'axes.labelsize': 15,
           'text.fontsize': 20,
           'legend.fontsize': 15,
           'xtick.labelsize': 18,
           'ytick.labelsize': 18,
           'text.usetex': True}
plt.rcParams.update(plotpar)
ocols = plot_colors()

def multilnlike(theta, x, y, yerr, p):
    theta = np.exp(theta)
    k = theta[0] * ExpSquaredKernel(theta[1]) * ExpSine2Kernel(theta[2], p)
    gp = george.GP(k)
    try:
        gp.compute(x, np.sqrt(theta[3]+yerr**2))
    except (ValueError, np.linalg.LinAlgError):
        return 10e25
    return -gp.lnlikelihood(y, quiet=True)

fname = 'all'
# load Wasp data
x1, y1, yerr1 = \
        np.genfromtxt('/Users/angusr/angusr/data/Wasp/1SWASPJ233549.28+002643.8_J233549_300_ORFG_TAMUZ.lc',
                      skip_header=110).T

# load MOST data
x2, y2, yerr2 = \
        np.genfromtxt("/Users/angusr/Downloads/267HIP1164542014reduced.dat").T

# load K2 data
x3, y3 = np.genfromtxt('/Users/angusr/angusr/data/Wasp/hip_mod.csv').T
yerr3 = np.ones_like(y3)*.0001

# y1 /= np.var(y1)  # variance normalised?
# y2 /= np.var(y2)  # variance normalised?
# y3 /= np.var(y3)  # variance normalised?
# y1 -= np.median(y1)  # subtract the median
# y2 -= np.median(y2)  # subtract the median
# y3 -= np.median(y3)  # subtract the median

x3 = x3 + 2454833 - 2450000
y3 -= np.median(y3)

print len(x1), len(x2), len(x3)

# plot data
plt.clf()
plt.subplot(3,1,1)
plt.errorbar(x1, y1, yerr=yerr1, fmt='k.', capsize=0, ecolor='.8')
plt.subplot(3,1,2)
plt.errorbar(x2, y2, yerr=yerr2, fmt='k.', capsize=0, ecolor='.8')
plt.subplot(3,1,3)
plt.errorbar(x3, y3, yerr=yerr3, fmt='k.', capsize=0, ecolor='.8')
plt.savefig('all_data')

# bin data?
bin_dat = False
x, y, yerr = x3, y3, yerr3
if bin_dat:
    x, y, yerr = bin_data(x3, y3, yerr3, min(x3), max(x3), .1)

theta = np.log([1e-6, 20. ** 2, 20, 1e-7]) # MOST init

# initial optimisation
p = 14.
before_and_after(theta, x, y, yerr, p, p, fname)  # all_baa.png

# grid over periods
periods = np.linspace(12, 17, 20)
# L, results = grid(theta, x, y, yerr, periods, neglnlike)
print neglnlike(theta, x, y, yerr, p)
