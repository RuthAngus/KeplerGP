import numpy as np
import pyfits
import matplotlib.pyplot as pl
from fixed_p_like import lnlike, predict, QP, neglnlike
import emcee
import triangle
from synth import synthetic_data, simple_s_data
import scipy.optimize as so
import time
from matplotlib.ticker import MaxNLocator
from scipy.optimize import fmin
from scipy.signal import periodogram, find_peaks_cwt, argrelextrema
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
    bi, pr = 200, 2000
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

def find_range(P, r, s):
    # Grid over periods
    if type(r)==tuple:
        mn, mx = r
    else:
#         mn, mx = P-(P*r), P+(P*r)
        mn, mx = P*r, P*(1./r)
    step = (mx-mn)/s
    return np.arange(mn, mx, step)

def global_max(x, y, yerr, theta, Periods, P, r, s, b, save):

        print "Initial parameters = (exp)", theta
        start = time.clock()
        print "Initial lnlike = ", -neglnlike(theta, x, y, yerr, P),"\n"
        elapsed = (time.clock() - start)
        print 'time =', elapsed

        L = np.zeros_like(Periods)
        results = np.zeros((len(L), 4))

        for i, p in enumerate(Periods):
            L[i], results[i,:] = maxlike(theta, x, y, yerr, p, i)

        iL = L==max(L)
        mlp = Periods[iL]
        mlresult = results[iL]
        print 'max likelihood period = ', mlp

        print 'mlresult = ', mlresult[0]
        print 'L', L
        # plot data
#         pl.clf()
#         pl.errorbar(x, y, yerr=yerr, fmt='k.', capsize=0, ecolor='0.5', zorder=2)
#         xs = np.linspace(min(x), max(x), 1000)
#         pl.plot(xs, predict(xs, x, y, yerr, mlresult[0], mlp)[0], color='#339999', linestyle = '-',\
#                 zorder=1, linewidth='2')
#         pl.xlabel('$\mathrm{Time~(days)}$')
#         pl.ylabel('$\mathrm{Normalised~Flux}$')
#         pl.gca().yaxis.set_major_locator(MaxNLocator(prune='lower'))
#         pl.savefig('/Users/angusr/Python/george/data/%sml_data%s' %(int(KID), save))

        # plot GPeriodogram
        pl.clf()
        area = sp.integrate.trapz(np.exp(L))
        pl.plot(Periods, (np.exp(L))/area, 'k-')
        pl.xlabel('Period')
        pl.ylabel('Likelihood')
        tdata = np.genfromtxt('/Users/angusr/angusr/Suz_simulations/final_table.txt', skip_header=1).T
        truemin = tdata[19][KID]
        truemax = tdata[20][KID]
        true = .5*(truemin+truemax)
        pl.title('Period = %s, true = %s, init = %s' %(mlp, true, P))
        pl.savefig('/Users/angusr/Python/george/likelihood/%shr_ml_likelihood%s' %(int(KID), save))

        # set period prior boundaries
        bm = mlp - b*mlp
        bp = mlp + b*mlp

        return L, mlp, bm, bp, mlresult[0]

def autocorrelation(x, y):

    # normalise so range is 2 - no idea if this is the right thing to do...
    y = 2*y/(max(y)-min(y))
    y = y-np.median(y)

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
    pl.plot(x[:4000], y[:4000], 'k.')
    pl.title('Period = %s' %period)
    pl.subplot(2,1,2)
    pl.plot(np.arange(5000)*cadence, acf[:5000])
#     pl.plot(np.arange(len(acf))*cadence, acf)
    [pl.axvline(peak*cadence, linestyle = '--', color = 'r') for peak in peaks]
    pl.xlim(0, 5000*cadence)
    pl.savefig('/Users/angusr/Python/george/acf/%shr_acf' %int(KID))
#     np.savetxt('/Users/angusr/Python/george/acf/%sacf_per.txt'%int(KID), period)

    return period

def pgram(y, r, P, highres):

    # calculate periodogram
    freq, power = periodogram(y)
    per = 1./freq

    # limit range
#     a = (per < P + (r*P)) * (per > P - (r*P))
    a = (0 < P ) * (P < 100)
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
            pl.savefig('/Users/angusr/Python/george/pgram/%shr_periodogram1' %int(KID))
        else:
            pl.savefig('Users/angusr/Python/george/pgram/%shr_periodogram2' %int(KID))
        np.savetxt('/Users/angusr/Python/george/pgram/%shr_pgram_per.txt', period)

        return period
    except:
        "ValueError:"
        pass

# subsample and truncate data
def subs(x, y, yerr, p_init, l):
    nday = 48. # number of data points/day
    npoints = 100. # number of data points needed/period
    subsamp = int(round(nday*p_init/npoints))
    if subsamp > 0:
        x = x[::subsamp]
        y = y[::subsamp]
        yerr = yerr[::subsamp]

    return x[:l], y[:l], yerr[:l]

def find_mins(L):
    # recalculate period range
    iL = np.where(L==max(L))[0]
    try:
        lmin = max(argrelextrema(L[:iL+1], np.less)[0])
    except:
        "max() arg is an empty sequence"
        lmin = 0
    try:
        rmin = int(min(argrelextrema(L[iL-1:], np.less)[0])+(iL-1))
    except:
        "min() arg is an empty sequence"
        rmin = -1
    return lmin, rmin

if __name__ == "__main__":

    cadence = 0.02043365

    KIDs = range(1004)
    strKIDs = []
    [strKIDs.append(str(KIDs[i]).zfill(4)) for i in KIDs]

    save_results = np.zeros((len(KIDs), 8))

    KIDs = range(1004)

    # load mean acf periods
    data = np.genfromtxt('/Users/angusr/Python/KeplerGP/rotation/acf_vs_true.txt').T
    p_init = data[2]

    star = 38
    k = star
    KID = star

    print 'star = ', KID, k

    p_init = 2.
    print 'p_init', p_init

    # Load light curves
    data = np.genfromtxt("/Users/angusr/angusr/Suz_simulations/final/lightcurve_%s.txt" \
            %strKIDs[k]).T
    x = data[0]
    y = data[1]
    yerr = y*1e-4 # one part per million #FIXME: this is made up!

    r = .4 # range of periods to try
    s = 100. # number of periods to try
    b = .2 # prior boundaries

    # normalise so range is 2 - no idea if this is the right thing to do...
    yerr = 2*yerr/(max(y)-min(y))
    y = 2*y/(max(y)-min(y))
    y = y-np.median(y)

    # subsample and truncate
    x_sub, y_sub, yerr_sub = subs(x, y, yerr, p_init, 500.)

    theta = [-2., -2., -1.2, 1.]

    # find range of periods to calculate L over
    Periods = find_range(p_init, r, s)

    # Find first global max
    L, mlp, bm, bp, mlh = global_max(x_sub, y_sub, yerr_sub, theta, Periods, p_init, \
            r, s, b, '1')

    np.savetxt('%shr_ml_results1.txt' %int(KID), np.transpose((Periods, L)))

    # find minima either side of peak
    #FIXME: maybe I should have a bit of leeway either side of the peak?
    lmin, rmin = find_mins(L)
    r = (Periods[lmin-2], Periods[rmin+2])
    Periods = find_range(mlp, r, s)

    # zoom in on highest peak
    L, mlp, bm, bp, mlh = global_max(x_sub, y_sub, yerr_sub, theta, Periods, p_init, \
            r, s, b, '2')

    np.savetxt('%shr_ml_results2.txt' %int(KID), np.transpose((Periods, L)))

    save_results[k,:] = np.array([KID, mlp[0], r[0], r[1], mlh[0], mlh[1], mlh[2], mlh[3]])
    print 'saving'
    print save_results
    np.savetxt('/Users/angusr/Python/george/inj_results/%shr_results.txt'%int(KID), save_results)
