import numpy as np
import matplotlib.pyplot as plt
import george
from george.kernels import ExpSine2Kernel, ExpSquaredKernel, WhiteKernel
from scipy.optimize import minimize, fmin
from colors import plot_colors

def multilnlike(theta, x1, x2, x3, x4, y1, y2, y3, y4,
                yerr1, yerr2, yerr3, yerr4, p):
    lnlike = []
    theta = np.exp(theta)
    k = theta[0] * ExpSquaredKernel(theta[1]) * ExpSine2Kernel(theta[2], p)
    gp = george.GP(k)
    try:
        gp.compute(x1, np.sqrt(theta[3]+yerr1**2))
    except (ValueError, np.linalg.LinAlgError):
        return 10e25
    lnlike.append(-gp.lnlikelihood(y1, quiet=True))
    try:
        gp.compute(x2, np.sqrt(theta[4]+yerr2**2))
    except (ValueError, np.linalg.LinAlgError):
        return 10e25
    lnlike.append(-gp.lnlikelihood(y2, quiet=True))
    try:
        gp.compute(x3, np.sqrt(theta[5]+yerr3**2))
    except (ValueError, np.linalg.LinAlgError):
        return 10e25
    lnlike.append(-gp.lnlikelihood(y3, quiet=True))
    try:
        gp.compute(x4, np.sqrt(theta[6]+yerr4**2))
    except (ValueError, np.linalg.LinAlgError):
        return 10e25
    lnlike.append(-gp.lnlikelihood(y4, quiet=True))
    return np.logaddexp.reduce(np.array(lnlike), axis=0)

def multilnlike_emcee_comb(theta, x, y, yerr):

    yerr[:121] = np.sqrt(theta[3]+yerr[:121]**2)
    yerr[122:304] = np.sqrt(theta[4]+yerr[122:304]**2)
    yerr[305:332] = np.sqrt(theta[5]+yerr[305:332]**2)
    yerr[333:] = np.sqrt(theta[6]+yerr[333:]**2)

    theta = np.exp(theta)
    k = theta[0] * ExpSquaredKernel(theta[1]) * ExpSine2Kernel(theta[2], theta[7])
    gp = george.GP(k)
    try:
        gp.compute(x, yerr)
    except (ValueError, np.linalg.LinAlgError):
        return 10e25
    return gp.lnlikelihood(y, quiet=True)

def multilnlike_emcee_wasp(theta, x1, x2, x3, x4, x5, y1, y2, y3, y4, y5,
                yerr1, yerr2, yerr3, yerr4, yerr5):
    lnlike = []
    theta = np.exp(theta)
    k = theta[0] * ExpSquaredKernel(theta[1]) * ExpSine2Kernel(theta[2], theta[7])
    gp = george.GP(k)
    try:
        gp.compute(x1, np.sqrt(theta[3]+yerr1**2))
    except (ValueError, np.linalg.LinAlgError):
        return 10e25
    lnlike.append(gp.lnlikelihood(y1, quiet=True))
    try:
        gp.compute(x2, np.sqrt(theta[4]+yerr2**2))
    except (ValueError, np.linalg.LinAlgError):
        return 10e25
    lnlike.append(gp.lnlikelihood(y2, quiet=True))
    try:
        gp.compute(x3, np.sqrt(theta[5]+yerr3**2))
    except (ValueError, np.linalg.LinAlgError):
        return 10e25
    lnlike.append(gp.lnlikelihood(y3, quiet=True))
    try:
        gp.compute(x4, np.sqrt(theta[6]+yerr4**2))
    except (ValueError, np.linalg.LinAlgError):
        return 10e25
    lnlike.append(gp.lnlikelihood(y4, quiet=True))
    try:
        gp.compute(x5, np.sqrt(theta[7]+yerr5**2))
    except (ValueError, np.linalg.LinAlgError):
        return 10e25
    lnlike.append(gp.lnlikelihood(y5, quiet=True))
    return np.logaddexp.reduce(np.array(lnlike), axis=0)

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

def before_and_after(theta, x, y, yerr, p1, p2, fname):
    xs = np.linspace(min(x), max(x), 100)
    ocols = plot_colors()

    # plot initial guess
    plt.subplot(2,1,1)
    plt.errorbar(x, y, yerr=yerr, fmt='k.', capsize=0, ecolor='.8')
    mu, cov = predict(theta, xs, x, y, yerr, p1)
    plt.plot(xs, mu, color=ocols.blue, label="$\mathrm{Period}=%s$" % p1)
    plt.xlabel('$\mathrm{Time~(days)}$')
    plt.ylabel('$\mathrm{RV~(ms}_{-1}\mathrm{)}$')
    plt.subplots_adjust(hspace=.2)
    plt.legend()

    # optimise hyperparameters
    new_theta = fmin(neglnlike, theta, args=(x, y, yerr, p2))

    # plot final guess
    plt.subplot(2,1,2)
    plt.errorbar(x, y, yerr=yerr, fmt='k.', capsize=0, ecolor='.8')
    mu, cov = predict(new_theta, xs, x, y, yerr, p2)
    plt.plot(xs, mu, color=ocols.orange, label="$\mathrm{Period}=%s$" % p2)
    plt.xlabel('$\mathrm{Time~(days)}$')
    plt.ylabel('$\mathrm{RV~(ms}_{-1}\mathrm{)}$')
    plt.subplots_adjust(hspace=.3)
    plt.legend()
    plt.savefig('%s_baa' % fname)

