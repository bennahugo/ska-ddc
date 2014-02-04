'''
Created on Feb 3, 2014

@author: benjamin
Computes the signal to noise ratio of two signals
See http://en.wikipedia.org/wiki/Signal-to-noise_ratio

Steps:
1. Computes the shift argmax(convolution between S and N) between 
the the noise and the tone and cancels out the phase shift using
the shift theorem.
http://www.dsprelated.com/dspbooks/mdft/Shift_Theorem.html
2. Computes the SNR (in dB, using the standard deviation, as shown in
the wikipedia article) 
'''
from scipy import *
import sys
import numpy as np
#import scipy.signal

if len(sys.argv) < 4:
    print "provide [REAL tone file], [REAL signal file], [offset to read from in signal file] and [optional: shift if known in advance]"
    sys.exit(1)
tone_file = sys.argv[1]
signal_file = sys.argv[2]
signal_start_offset = int(sys.argv[3])
tone = np.fromfile(tone_file,dtype=np.int8).astype(np.float32)
signal = np.fromfile(signal_file,dtype=np.int8)[signal_start_offset:].astype(np.float32)

def pad(signal,new_length):
    assert(new_length > len(signal)),"cannot cut away data"
    out = np.zeros(new_length)
    out[0:len(signal)] = signal
    return out

def fastRealXCorrelate(signal1,signal2):
    '''
    By the convolution theorem (we assume the signals are real, so we do not compute the conjugate here)
    '''
    new_length = len(signal1) + len(signal2) - 1
    padded_s1 = pad(signal1,new_length)
    padded_s2 = pad(signal2,new_length)
    return np.real(np.fft.fftshift(np.fft.ifft(np.fft.fft(padded_s1) * np.fft.fft(padded_s2[::-1]))))

def shiftSignal(signal,time_steps):
    '''
    By the shift theorem:
    '''
    shift_exp = exp(-2 * np.pi * 1.0j * time_steps * np.arange(0,len(signal))/len(signal))
    return np.real(np.fft.ifft(np.fft.fft(signal) * shift_exp))
        
def argMax(arr):
    maxV = arr[0]
    maxA = 0
    for t in range(1,len(arr)):
        if arr[t] > maxV:
            maxV = arr[t]
            maxA = t
    return maxA 

def snr(signal,noise):
    stdev_sig = np.std(signal[0:max(len(signal),len(noise))])
    stdev_noise = np.std(noise[0:max(len(signal),len(noise))])
    return 10 * log10(stdev_sig/stdev_noise)
computed_shift = 0
if len(sys.argv) < 5:
	xCorFast = fastRealXCorrelate(tone,signal)
	computed_shift = (len(tone) + len(signal) - 1) - (argMax(xCorFast) + 1)
else:
	computed_shift = int(sys.argv[4])
print "Shifted by %d elements" % (computed_shift)
signal_inv_shifted = shiftSignal(signal,-computed_shift)
noise = signal_inv_shifted - tone[:len(tone) - signal_start_offset]
tone_to_noise = snr(tone[:len(tone) - signal_start_offset],noise)
print "Signal to noise (input tone to noise): %f dB" % tone_to_noise

