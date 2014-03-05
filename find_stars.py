
import numpy as np
import pylab as p
import load_data
import fitConsecutive_simple
import embedfilt


def detrend():

    # Load data.
    time, flux, cadence = load_data.load('012317678', quarter = 3)
    flux = flux[:100]; time = time[:100]

    # Initial processing: filtdet
    # Filter data using a sub-space filter with 20 embeddings

   

    # p.close(1)
    # p.figure(1)
    # p.subplot(3,1,1)
    # p.plot(time, flux, 'k.')

    flux = embedfilt.embedfilt(flux)
    print len(time), len(flux)
    
    # p.subplot(3,1,2)
    # p.plot(time[1:], flux[1:], 'k.')
    raw_input('enter')
    
    # remove linear trends
    ply = np.polyfit(time, flux, 1)
    rmv = ply[0] + ply[1]*time 
    flux /= rmv
    
    p.subplot(3,1,3)
    p.plot(time[1:], flux[1:], 'k.')
    raw_input('enter')
    
    # join data together (last point in the last data series matches the first point in the next)
    
    # First GP
    time1, flux1 = fitConsecutive_simple.fitGP(time[:500], flux[:500])
    p.close(5)
    p.figure(5)
    p.plot(time1[100:-100], flux1[100:-100], 'k.')
    raw_input('enter')
    
    # Use maximum likelihood hyperparams from first GP as initial values in second.

    # Second GP
    time2, flux2 = fitConsecutive_simple.fitGP(time1, flux1)
    p.close(6)
    p.figure(6)
    p.plot(time2, flux2, 'k.')
    
    # BLS search for planets within residuals

    # Still to do: intial sub-space filtering and linear detrend
    # Speed up inversion
    # Join quarters
    # incorporate Toeplitz code?
    # Add BLS what ideas does Steve have for this?
    # Test Dona's code 
    # Cut out exponential trends - preceeded by gaps.
    # Rational quadratic kernel?
