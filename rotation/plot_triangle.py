import numpy as np
import matplotlib.pyplot as pl
import triangle
import h5py

def open_samples(fname):
    with h5py.File("samples_%s" %fname, "r") as f:
        samples = f["samples"][:, 50:, :]
    nwalkers, n, ndim = samples.shape
    return samples.reshape((-1, ndim))

def result(fname, samples):
    mcmc_result = map(lambda v: (v[1], v[2]-v[1], v[1]-v[0]),
                      zip(*np.percentile(samples, [16, 50, 84], axis=0)))
    np.savetxt("parameters%s.txt" %fname, np.array(mcmc_result))
    return mcmc_result

def triangle_plot(fname, mcmc_result, samples, fig_labels):
    mres = np.array(mcmc_result)[:, 0]
    print 'mcmc_result = ', mres
    fig = triangle.corner(samples, truths=mres, labels=fig_labels)
    fig.savefig("triangle_%s.png" % fname)

def trace_plot(fname, samples, ndim):
    pl.figure()
    for i in range(ndim):
        pl.clf()
        pl.plot(samples[:, :, i].T, 'k-', alpha=0.3)
        pl.savefig("%s%s.png" %(i, fname))

if __name__ == "__main__":
    fname = 'MOST'

    samples = open_samples(fname)
    mcmc_result = result(fname, samples)

    fig_labels = ["$A$", "$l1$", "$l2$", "$wn$", "$P$"]
    triangle_plot(fname, mcmc_result, samples, fig_labels)

    ndim = len(fig_labels)
    trace_plot(fname, samples, ndim)
