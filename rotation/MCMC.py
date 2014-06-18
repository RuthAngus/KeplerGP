import numpy as np
import matplotlib.pyplot as pl
import emcee
import triangle
from injection_tests import MCMC, subs

# for KID in range(1000):
KID = 34

# load optimised hyperparameters
# KID, period, r1, r0, m0, m1, m2, m3
data = np.genfromtxt('/Users/angusr/Python/george/inj_results/%sresults.txt' \
        %int(KID)).T
mlp = data[1][KID]
m0 = data[4][KID]
m1 = data[5][KID]
m2 = data[6][KID]
m3 = data[7][KID]
print 'mlp', mlp
raw_input('enter')

# load profile likelihood surface
data = np.genfromtxt('/Users/angusr/Python/george/%sml_results2.txt' \
        %int(KID)).T

print 'star = ', KID
p = data[0]
L = data[1]

#     r0 = data[2][0]
#     r1 = data[3][0]
r0, r1 = min(p), max(p)
#     m = [data[4][0], data[5][0], data[6][0], data[7][0], data[1][0]]
m = [m0, m1, m2, m3, mlp]
print 'm = ', m
print 'r0, r1 = ', r0, r1
raw_input('enter')

# load lightcurve
# Load light curves
data = np.genfromtxt("/Users/angusr/angusr/Suz_simulations/final/lightcurve_00%s.txt" \
        %KID).T
x = data[0]
y = data[1]
yerr = y*2e-5 # one part per million #FIXME: this is made up!

# normalise so range is 2 - no idea if this is the right thing to do...
yerr = 2*yerr/(max(y)-min(y))
y = 2*y/(max(y)-min(y))
y = y-np.median(y)

print 'subsample and truncate'
x_sub, y_sub, yerr_sub = subs(x, y, yerr, mlp, 500.)

MCMC(m, x_sub, y_sub, yerr_sub, r0, r1)
raw_input('enter')

#     m = np.empty(len(theta)+1)
#     m[:len(theta)] = theta
#     m[-1] = mlp
#     MCMC(m, x, y, yerr, r[0], r[1])
