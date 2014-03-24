import numpy as np
import pyfits
import matplotlib.pyplot as pl
from lnlikefn import lnlike

# Load data
hdulist = pyfits.open("/Users/angusr/angusr/data2/Q3_public/kplr003223000-2009350155506_llc.fits")
tbdata = hdulist[1].data
x = tbdata["TIME"]
y = tbdata["PDCSAP_FLUX"]
yerr = tbdata["PDCSAP_FLUX_ERR"]
q = tbdata["SAP_QUALITY"]

# remove nans
n = np.isfinite(x)*np.isfinite(y)*np.isfinite(yerr)*(q==0)
l = 500.
x = x[n][:l]
y = y[n][:l]
yerr = yerr[n][:l]
mu = np.median(y)
y = y/mu - 1.
yerr /= mu

pl.clf()
pl.errorbar(x, y, yerr=yerr, fmt='k.')
pl.xlabel('time (days)')
pl.savefig('data')

theta = [1e-8, 1., 10., 2.]
P = np.arange(0.1, 5, 0.1)
L = np.empty_like(P)

for i, per in enumerate(P):
    theta[1] = per
    L[i] = lnlike(theta, x, y, yerr)

pl.clf()
pl.plot(P, L, 'k-')
pl.xlabel('Period (days)')
pl.ylabel('likelihood')
pl.savefig('likelihood2')
