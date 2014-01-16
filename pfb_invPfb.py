reated on Jan 16, 2014

@author: benjamin
'''
from pylab import *
import sys
import scipy.signal

tone_freq = [i * 1e6 for i in [110]]
sampling_freq = 800e6

N = 256 #no. FFT samples
P = 8 #no. subfilters in the bank
window_length = N * P #length of the hamming window

no_samples = window_length * 1 #number of samples should hopefully be a multiple of the window length  
pad = N*P #pad the tone


'''
generate a fake tone and pad with N*P zeros
'''
tone = np.zeros(no_samples + pad) 
for f in tone_freq:
    for n in np.arange(0,no_samples):
        tone[n] += np.sin(2 * np.pi * n * (f / float(sampling_freq)))
    
'''
generate the hamming window for leakage suppression
'''
w = np.hamming(window_length)

'''
generate the inverse hamming window for inverse pfb
'''
w_inverse = np.fliplr(w.reshape(N,P)).reshape(window_length) 

'''
 "filter" N * P elements from the tone at a time, stepping through the tone by N samples
'''
pad = N*P
pfb_output = np.zeros(no_samples).astype(np.complex64)

for lB in range(0,no_samples,N):
    w_out = tone[lB:(lB + window_length)] * w #element-wise multiply N*P samples with the window function
    #element-wise accumulate each of the subfilters into one block 
    for n in range(0,N):
        accum = 0
        for blockID in range(0,P):
            accum += w_out[(n + blockID*N)]
        pfb_output[lB+n] = accum  
    pfb_output[lB:lB+N] = np.fft.fft(pfb_output[lB:lB+N])

'''
Plot
'''
figure(1)
subplot(211)
title("Tone")
plot(tone)
subplot(212)
plot(np.arange(0,sampling_freq,sampling_freq/no_samples) / 1.0e6,np.abs(np.imag(np.fft.fft(tone[0:no_samples]))))
figure(2)
subplot(211)
title("Hamming window")
plot(w)
subplot(212)
title("Hamming window with subfilters flipped")
plot(w_inverse)

figure(3)
title("PFB output")
plot(np.imag(pfb_output))
show()
