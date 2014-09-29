import numpy as np
import matplotlib.pyplot as plt
import george
from george.kernels import ExpSine2Kernel, ExpSquaredKernel, WhiteKernel
from scipy.optimize import minimize, fmin

t = 100 # limit number of points
data = np.genfromtxt('/Users/angusr/angusr/data/Wasp/1SWASPJ233549.28+002643.8_J233549_300_ORFG_TAMUZ.lc', skip_header=110).T[:,:t]

x = data[0]
y = data[1] - np.median(data[1])
yerr = data[2]

def predict(theta, xs, x, y, yerr, p):
    k = theta[0] * ExpSquaredKernel(theta[1]) * ExpSine2Kernel(theta[2], p)
    gp = george.GP(k)
    gp.compute(x, np.sqrt(theta[3]+yerr**2))
    return gp.predict(y, xs)

# def neglnlike(theta, x, y, yerr, p):
def neglnlike(theta, x, y, yerr):
    k = theta[0] * ExpSquaredKernel(theta[1]) * ExpSine2Kernel(theta[2], theta[4])
    gp = george.GP(k)
    try:
        gp.compute(x, np.sqrt(theta[3]+yerr**2))
    except (ValueError, np.linalg.LinAlgError):
        return 10e25
    return -gp.lnlikelihood(y, quiet=True)

# initial guess
# theta = [1.**2, .5 ** 2, 100., 0.05]
theta = [1.**2, .5 ** 2, 100., 0.05, 16.]
xs = np.linspace(min(x), max(x), 100)

k = theta[0] * ExpSquaredKernel(theta[1]) * ExpSine2Kernel(theta[2], theta[4])
k += WhiteKernel(theta[3])
gp = george.GP(k)

periods = np.linspace(3., 30, 20)

# print neglnlike(theta, x, y, yerr, periods[0]), 'nll'
print neglnlike(theta, x, y, yerr), 'nll'
result = gp.optimize(x, y, yerr)
print result[0]
# print neglnlike(result[0], x, y, yerr, periods[0]), 'nll'
print neglnlike(result[0], x, y, yerr), 'nll'

L = np.empty_like(periods)
thetas = np.zeros((len(theta), len(periods)))

for i, p in enumerate(periods):

    k = theta[0] * ExpSquaredKernel(theta[1]) * ExpSine2Kernel(theta[2], p)
    k += WhiteKernel(theta[3])
    gp = george.GP(k)

#     result = gp.optimize(x, y, yerr, dims=[0,1,2,3])
    result = gp.optimize(x, y, yerr)

    thetas[:,i] = result[0]

    print result[0]
    L[i] = neglnlike(result[0], x, y, yerr, p)

    print 'Period = ', p, 'likelihood = ', L[i]
    raw_input('enter')

plt.clf()
plt.plot(periods, L)
plt.savefig('Wasp_likes')

ml = max(L)
l = L==ml
mp = periods[l]
mtheta = thetas[:,l]
print mtheta
raw_input('enter')

ys = predict(mtheta, xs, x, y, yerr, mp)
plt.clf()
plt.errorbar(x, y, yerr=yerr, fmt='k.', capsize=0)
plt.plot(xs, ys[0], 'r')
plt.savefig('wasp')
