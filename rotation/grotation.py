import numpy as np
import matplotlib.pyplot as pl
import george
from george.kernels import ExpSquaredKernel
import emcee
import triangle

plotpar = {'axes.labelsize': 16,
           'text.fontsize': 16,
           'legend.fontsize': 14,
           'xtick.labelsize': 17,
           'ytick.labelsize': 17,
           'text.usetex': True}
pl.rcParams.update(plotpar)

# Load data
hdulist = pyfits.open("/Users/angusr/angusr/data2/Q3_public/kplr003223000-2009350155506_llc.fits")
tbdata = hdulist[1].data
x = tbdata["TIME"]
y = tbdata["PDCSAP_FLUX"]
yerr = tbdata["PDCSAP_FLUX_ERR"]

# remove nans
n = np.isfinite(x)*np.isfinite(y)*np.isfinite(yerr)
l = 500.
x = x[n][:l]
y = y[n][:l]
yerr = yerr[n][:l]

# subsample
xplot = x
yplot = y/np.mean(y) - 1
yerrplot = yerr
subsamp = 5
x = x[0:-1:subsamp]
y = y[0:-1:subsamp]
yerr = yerr[0:-1:subsamp]

# test data
xs = np.r_[min(x)-1:max(x)+1:500j]

# format data
y = np.array(y)/np.mean(y) - 1.
X = np.matrix([x]).T # convert inputs to matrix form (N x D)
Xs = np.matrix([xs]).T  # convert inputs to matrix form (Q x D)

# Set up the Gaussian process.
kernel = ExpSquaredKernel(1.0)
gp = george.GaussianProcess(kernel)

# Pre-compute the factorization of the matrix.
gp.compute(x, yerr)

# Compute the log likelihood.
print(gp.lnlikelihood(y))

# Draw 100 samples from the predictive conditional distribution.
t = np.linspace(0, 10, 500)
samples = gp.sample_conditional(y, t, size=100)
pl.clf()
pl.plot(x, y, 'k.')
pl.plot(t, samples[0], 'k')
pl.savefig('george')


# load data
