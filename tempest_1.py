
from IPython.display import clear_output
import numpy as np
import scipy.signal as sg
from scipy.fft import fft, ifft
import matplotlib.pyplot as plt
from PIL import Image
import math
import numpy as np

# Load the binary file
sig = np.fromfile('/users/elo/myate/SNUM3/TempestSDR_runtimeApplication/dumpIQ_0.dat', dtype=np.csingle) #float complex
Fs =20e6
Ts = 1 / Fs


def mag2db(sig):
# Simple magnitude to Decibel convertion using array ... comprehension 
    return [10*math.log10(x) for x in sig]

def autocorr_circ(sig):
    X = np.fft.fft(sig)
    X_conj = np.conj(X) 
    y = np.fft.ifft(X * X_conj)  # Autocorrelation of the input signal

    return np.abs(y)


lags = (np.arange(len(autocorr_circ(sig))) / Fs)*1000


def delay_to_rate(delay):
    return 1 / delay

def rate_to_delay(rate):
    return 1 / rate

def delay_to_index(delay):
    return delay*Fs

 #demodulation 
sig_abs = np.abs(sig)

def spectralAnalysis(sig):
    """ Simple spectral analysis based on Wiener-Kinchine ...theorem """
    # PSD calculation
    c = sg.correlate(sig, sig, mode='same') # ...Autocorrelation of the input signal
    y = np.abs(fft(c))**2 # square modulus of the fourier ...transform of the autocorr
    Y = mag2db(y) # In dB
    # PSD plot
    fig, ax = plt.subplots()
    plt.plot(Y)
    ax.set_xlabel('Lag in seconds')
    ax.set_ylabel('Correlation magnitude') #At this end of this part

spectralAnalysis(sig)


def spectralAnalysis_correlate_sec(sig,Fs):
    y = autocorr_circ(sig)  # Autocorrelation of the input signal
    # PSD plot
    plt.figure
    plt.plot(lags, y)
    plt.xlabel('Lag in second')
    plt.ylabel('Correlation magnitude')
    plt.show()
    

#3.3 spectralAnalysis(sig)
spectralAnalysis_correlate_sec(sig,Fs)
print(" len(sig) = " , len(sig))

#3.4 calcul du index de retard

#5 40Hz et 100Hz
#for 40Hz 
delay_40Hz  = rate_to_delay(40)
delay_100Hz = rate_to_delay(100)
print(" delay 40 Hz" , delay_40Hz)
print(" delay 100 Hz" ,delay_100Hz)

def find_subvector_zoom(sig , min, max):
    delay_min = rate_to_delay(min)*1000
    delay_max = rate_to_delay(max)*1000

    indice_zoom         = np.where((lags >= delay_max) & (lags <= delay_min))[0]
    subvector_auto      = autocorr_circ( sig )
    subvector_auto_zoom = subvector_auto[indice_zoom]
    subvector_lags_zoom = lags[indice_zoom]
    print(subvector_auto_zoom)

    plt.figure
    plt.plot( subvector_lags_zoom ,subvector_auto_zoom )
    plt.xlabel('Lag in milliseconds')
    plt.ylabel('autocorrelation zoomer sur min et max')
    plt.show()

    max = np.max(subvector_auto_zoom)
    max_index = np.argmax(subvector_auto_zoom)
    print("max_index" , max_index)

    return max
print(" max entre 40Hz et 100Hz  = ",find_subvector_zoom(sig, 40, 100))

# 3.3 Find the screen configuration
def zoom_0_40us(sig) : 
    index_0    = int(delay_to_index(0))
    index_40us = int(delay_to_index(40e-6))

    print("indice 0s" , index_0)
    print("indice 40us" , index_40us)
    
    subvector_auto       = autocorr_circ( sig )
    subvector_auto_zoom  = subvector_auto[ index_0 : index_40us]
    subvector_lags_zoom  = lags[index_0 : index_40us]
    
    plt.figure(3)
    plt.plot( subvector_lags_zoom ,subvector_auto_zoom )
    plt.xlabel('Lag in milliseconds')
    plt.ylabel('autocorrelation zoomer sur min et max')
    plt.show()

    return subvector_auto_zoom

zoom_0_40us(sig)
    









