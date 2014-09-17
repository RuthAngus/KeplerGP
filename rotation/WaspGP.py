
# coding: utf-8

# In[7]:

import numpy as np
import matplotlib.pyplot as plt
get_ipython().magic(u'matplotlib inline')
import george
from george.kernels import ExpSine2Kernel, ExpSquaredKernel


# In[8]:

data = np.genfromtxt('/Users/angusr/angusr/data/Wasp/1SWASPJ233549.28+002643.8_J233549_300_ORFG_TAMUZ.lc', skip_header=110).T
x = data[0]
y = data[1] - np.median(data[1])
yerr = data[2]


# In[9]:

plt.errorbar(x, y, yerr=yerr, fmt='k.', capsize=0)


# In[10]:

def predict(theta, xs, x, y, yerr):
    k = theta[0] * ExpSquaredKernel(theta[1])  # * ExpSine2Kernel(theta[2], theta[3])
    gp = george.GP(k)
    gp.compute(x, np.sqrt(theta[2]+yerr**2))
    return gp.predict(y, xs)


# In[6]:

theta = [1.**2, .5 ** 2, 0.05]
xs = np.linspace(min(x), max(x), 100)
ys = predict(theta, xs, x, y, yerr)


# In[ ]:

plt.errorbar(x, y, yerr=yerr, fmt='k.')
plt.plot(xs, ys[0], 'r')


# In[ ]:



