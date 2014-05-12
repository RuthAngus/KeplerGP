import numpy as np
import pyfits
import matplotlib.pyplot as pl
from fixed_p_like import lnlike, predict, QP, neglnlike
import emcee
import triangle
from load_dataGP import load
from synth import synthetic_data, simple_s_data
import scipy.optimize as so
import time
from matplotlib.ticker import MaxNLocator
from scipy.optimize import fmin
from scipy.signal import periodogram, find_peaks_cwt
import scipy as sp
import pylab as pb

ocols = ['#FF9933','#66CCCC' , '#FF33CC', '#3399FF', '#CC0066', '#99CC99', '#9933FF', '#CC0000']
plotpar = {'axes.labelsize': 20,
           'text.fontsize': 20,
           'legend.fontsize': 15,
           'xtick.labelsize': 18,
           'ytick.labelsize': 18,
           'text.usetex': True}
pl.rcParams.update(plotpar)

# flat priors (quasi-periodic)
def lnprior(theta, bm, bp):
    if -16.<theta[0]<16. and -16.<theta[1]<16. and -16.<theta[2]<16.\
            and -16.<theta[3]<16. and bm<theta[4]<bp:
        return 0.0
    return -np.inf

# posterior prob
def lnprob(theta, x, y, yerr, bm, bp):
    lp = lnprior(theta, bm, bp)
    if not np.isfinite(lp):
        return -np.inf
    try:
        return lp + lnlike(theta, x, y, yerr)
    except:
        print theta
        raise

# run the mcmc over a range of period values
def maxlike(theta, x, y, yerr, P, name):

    print 'initial values = ', theta, P
    print 'initial nll = ', neglnlike(theta, x, y, yerr, P)
    result = fmin(neglnlike, theta, args = (x, y, yerr, P))
    print 'final values = ', result, P
    like = neglnlike(result, x, y, yerr, P)
    print 'final likelihood = ', like

#     savedata = np.empty(len(result)+2)
#     savedata[:len(result)] = result
#     savedata[-2] = P
#     savedata[-1] = like
#     np.savetxt('results/ml_result%s.txt'%name, savedata)

    return -like, result

def MCMC(theta, x, y, yerr, bm, bp):

    # Sample the posterior probability for m.
    nwalkers, ndim = 64, len(theta)
    p0 = [theta+1e-4*np.random.rand(ndim) for i in range(nwalkers)]
    sampler = emcee.EnsembleSampler(nwalkers, ndim, lnprob, args = (x, y, yerr, bm, bp))
    bi, pr = 200, 800
    start = time.clock()
    print("Burn-in")
    p0, lp, state = sampler.run_mcmc(p0, bi)
    sampler.reset()
    print("Production run")
    sampler.run_mcmc(p0, pr)
    elapsed = time.clock() - start
    print 'time = ', elapsed/60., 'mins'

    print("Making triangle plots")
    fig_labels = ["$A$", "$l_2$", "$l_1$", "$s$", "$P$"]
    fig = triangle.corner(sampler.flatchain, truths=theta, labels=fig_labels[:len(theta)])
    fig.savefig("triangle.png")

    print("Plotting traces")
    pl.figure()
    for i in range(ndim):
        pl.clf()
        pl.axhline(theta[i], color = "r", zorder=2)
        pl.plot(sampler.chain[:, :, i].T, 'k-', alpha=0.3, zorder=1)
        pl.savefig("{0}.png".format(i))

    # Flatten chain
    samples = sampler.chain[:, 50:, :].reshape((-1, ndim))

    # Find values
    mcmc_result = map(lambda v: (v[1], v[2]-v[1], v[1]-v[0]),
                      zip(*np.percentile(samples, [16, 50, 84], axis=0)))

    theta = np.array(mcmc_result)[:, 0]
    print 'mcmc result = ', theta

    like = lnlike(theta, x, y, yerr)
    print "Final lnlike = ", like

    # plot mcmc result
    pl.clf()
    pl.errorbar(x, y, yerr=yerr, fmt='k.')
    xs = np.arange(min(x), max(x), 0.01)
    pl.plot(xs, predict(xs, x, y, yerr, theta, theta[4])[0], 'r-')
    pl.xlabel('time (days)')
    pl.savefig('result')

def global_max(x, y, yerr, theta, P, r, s, b):

        print "Initial parameters = (exp)", theta
        start = time.clock()
        print "Initial lnlike = ", -neglnlike(theta, x, y, yerr, P),"\n"
        elapsed = (time.clock() - start)
        print 'time =', elapsed

        # Grid over periods
        mn, mx = P-(P*r), P+(P*r)
        step = (mx-mn)/s
        Periods = np.arange(mn, mx, step)
        L = np.zeros_like(Periods)
        results = np.zeros((len(L), 4))

        for i, p in enumerate(Periods):
            L[i], results[i,:] = maxlike(theta, x, y, yerr, p, i)

        np.savetxt('%sml_results.txt' %int(KID), np.transpose((Periods, L)))

        mlp = Periods[L == max(L)]
        mlresult = results[L == max(L)]
        print 'max likelihood period = ', mlp

        print 'mlresult = ', mlresult[0]
        # plot data
        pl.clf()
        pl.errorbar(x, y, yerr=yerr, fmt='k.', capsize=0, ecolor='0.5', zorder=2)
        xs = np.linspace(min(x), max(x), 1000)
        pl.plot(xs, predict(xs, x, y, yerr, mlresult[0], mlp)[0], color='#339999', linestyle = '-',\
                zorder=1, linewidth='2')
        pl.xlabel('$\mathrm{Time~(days)}$')
        pl.ylabel('$\mathrm{Normalised~Flux}$')
        pl.gca().yaxis.set_major_locator(MaxNLocator(prune='lower'))
        pl.savefig('%sml_data' %int(KID))

        # plot GPeriodogram
        pl.clf()
        area = sp.integrate.trapz(np.exp(L))
        pl.plot(Periods, (np.exp(L))/area, 'k-')
        pl.xlabel('Period')
        pl.ylabel('Likelihood')
        pl.title('Period = %s' %mlp)
        pl.savefig('%sml_likelihood' %int(KID))

        # set period prior boundaries
        bm = mlp - b*mlp
        bp = mlp + b*mlp

        return mlp, bm, bp

def autocorrelation(x, y):

    # calculate autocorrelation fn
    lags, acf, lines, axis = pb.acorr(y, maxlags = len(y)/2.)

    # halve acf and find peaks
    acf = acf[len(acf)/2.:]
    pks = find_peaks_cwt(acf, np.arange(10, 20)) # play with these params,
    #they define peak widths
    peaks = pks[1:] # lose the first peak
    period =  peaks[0]*cadence
    print 'acf period = ', period

    pl.clf()
    pl.subplot(2,1,1)
    pl.plot(x, y, 'k.')
    pl.title('Period = %s' %period)
    pl.subplot(2,1,2)
    pl.plot(np.arange(len(acf))*cadence, acf)
    [pl.axvline(peak*cadence, linestyle = '--', color = 'r') for peak in peaks]
    pl.savefig('%sacf' %int(KID))

    return period

def pgram(y, r, P, highres):

    # calculate periodogram
    freq, power = periodogram(y)
    per = 1./freq

    # limit range
    a = (per < P + (r*P)) * (per > P - (r*P))
    per = per[a]
    power = power[a]

    # find peaks
#     pks = find_peaks_cwt(power, np.arange(.01, 10, .01))
    try:
        pks = find_peaks_cwt(power, np.arange(1, 10))
        per_peaks = per[pks][::-1]
        pow_peaks = power[pks][::-1]
        period = per_peaks[pow_peaks == max(pow_peaks)]
        print 'periodogram period = ', period

        # plot
        pl.clf()
        pl.plot(per, power)
        pl.xlim(P-(r*P), P+(r*P))
    #     pl.ylim(0, 100000)
        [pl.axvline(pk, linestyle = '--', color = 'r') for pk in per_peaks]
        pl.title('Period = %s' %period)
        if highres == True:
            pl.savefig('%speriodogram1' %int(KID))
        else:
            pl.savefig('%speriodogram2' %int(KID))

        return period
    except:
        "ValueError:"
        pass

if __name__ == "__main__":

    cadence = 0.02043365

    # Load target list with ACF periods
    data = np.genfromtxt('/Users/angusr/Python/george/targets.txt').T
    KIDs = data[0]
    p_init = data[1]

#     KIDs = KIDs[1:]
#     p_init = p_init[1:]

    for k, KID in enumerate(KIDs):

        print k, KID

        # Load real quarter 3 data
        x, y, yerr = load("/Users/angusr/angusr/data2/Q3_public/kplr0%s-2009350155506_llc.fits" %int(KID))

        r = .4 # range of periods to try
        s = 10. # number of periods to try
        b = .2 # prior boundaries

        pgram(y, r, p_init[k], highres = True)

        # normalise so range is 2 - no idea if this is the right thing to do...
        yerr = 2*yerr/(max(y)-min(y))
        y = 2*y/(max(y)-min(y))
        y = y-np.median(y)

        autocorrelation(x, y)

        # subsample and truncate data
        nday = 48. # number of data points/day
        npoints = 100. # number of data points needed/period
        subsamp = int(round(nday*p_init[k]/npoints))
        if subsamp > 0:
            x = x[::subsamp]
            y = y[::subsamp]
            yerr = yerr[::subsamp]

        l = 500. # truncate to 500 data points, total
        x = x[:l]
        y = y[:l]
        yerr = yerr[:l]

#         theta, P = [-2., -2., -1.2, 1.], 1.7 # generating fake data
        theta = [-2., -2., -1.2, 1.]

#         # generate fake data
#         K = QP(theta, x, yerr, P)
#         y = np.random.multivariate_normal(np.zeros(len(x)), K)

#         raw_input('enter')

        # THIS NEEDS TO RETURN ALL MAX LIKELIHOOD PARAMS
        mlp, bm, bp = global_max(x, y, yerr, theta, p_init[k], r, s, b)

#         # running MCMC over maxlikelihood period
#         m = np.empty(len(theta)+1)
#         m[:len(theta)] = theta
#         m[-1] = mlp
# #         m = [-2.16545599, -0.12804739, -0.19512039, 3.4567929, 2.496]
# #         bm, bp = 2., 3.
#         MCMC(m, x, y, yerr, bm, bp)
