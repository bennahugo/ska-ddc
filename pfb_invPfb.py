#!/bin/usr/python
'''
Created on Jan 16, 2014

@author: benjamin
'''
from pylab import *
import sys
import scipy.signal

sampling_freq = 800e6
max_freq = 0.5 * sampling_freq 
N = 512 #no. FFT samples
P = 8 #no. subfilters in the bank
window_length = N * P #length of the hamming window
no_bins = N / 2 #real FFTs mirror half the bins
#tone_freq = [max_freq / float(no_bins) * (100)]
#tone_freq = [max_freq / float(no_bins) * (150.3)]
#tone_freq = [max_freq / float(no_bins) * (100),max_freq / float(no_bins) * (150)] 
tone_freq = [max_freq / float(no_bins) * (100),max_freq / float(no_bins) * (150.45)] 
no_samples = window_length*6 #number of samples should hopefully be a multiple of the window length  
pad = N*P #pad the tone

'''
generate a fake tone
'''
tone = np.zeros(no_samples) 
for f in tone_freq:
    for n in np.arange(0,no_samples):
        tone[n] += np.sin(2 * np.pi * n * (f / float(sampling_freq)))

for f in tone_freq:
    print "frequency %f MHz should be in channel %f after filtering" % (f / float(1e6),f / (max_freq / float(no_bins)))

'''
FORWARD PFB
'''
#setup the windowing function
w = scipy.signal.firwin(P * N, 1. / N).reshape(P, N)

#filter with the filterbank
pfb_input = np.zeros(no_samples + pad).astype(np.complex64)

pfb_input[pad:pad+no_samples] = tone
pfb_filtered_output = np.zeros(no_samples).astype(np.complex64)
pfb_output = np.zeros(no_samples).astype(np.complex64)
for lB in range(0,no_samples,N):
    pfb_filtered_output[lB:lB+N] = ((pfb_input[lB:lB+(P*N)].reshape(P,N))*w).sum(axis=0) 
    pfb_output[lB:lB+N] = np.fft.fft(pfb_filtered_output[lB:lB+N])
      

'''
take Short Time FFTs without filtering for comparison
'''
unfiltered_output = np.zeros(no_samples).astype(np.complex64)
for lB in range(0,no_samples,N):
    unfiltered_output[lB:lB+N] = np.fft.fft(pfb_input[lB:lB+N])
    
    
'''
INVERSE PFB
'''
#setup the inverse windowing function
#w_i = (scipy.signal.firwin(P * N, 1. / N)[::-1]).reshape(P, N)
w_i = w # filter is semetrical, so there should not be a difference if we flip it or not


len_of_valid_pfb_output = no_samples - pad
pfb_inverse_input = np.zeros(len_of_valid_pfb_output + pad).astype(np.complex64)
pfb_inverse_input[pad:pad+len_of_valid_pfb_output] = pfb_output[pad:no_samples]
pfb_inverse_ifft_output = np.zeros(len_of_valid_pfb_output + pad).astype(np.complex64)
pfb_inverse_output = np.zeros(len_of_valid_pfb_output).astype(np.complex64)

'''
for computational efficiency invert every FFT from the forward process only once... for xx large data
we'll need a ring buffer to store the previous IFFTs
'''

for lB in range(0,no_samples,N):
    pfb_inverse_ifft_output[lB:lB+N] = np.fft.ifft(pfb_inverse_input[lB:lB+N])

for lB in range(0,len_of_valid_pfb_output,N):
    pfb_inverse_output[lB:lB+N] = np.fliplr(pfb_inverse_ifft_output[lB:lB+(P*N)].reshape(P,N)*w_i).sum(axis=0)

xCor = scipy.signal.correlate(pfb_inverse_output[pad:len_of_valid_pfb_output], tone[0:len_of_valid_pfb_output-pad])


'''
Plot
'''
figure(1)
subplot(211)
title("Unfiltered Short time fast fourier transforms")
plot(pfb_input)
axvline(x=pad,linewidth=2, color='r')
for x in range(N, no_samples-pad,N):
    axvline(x=pad + x,linewidth=1, color='g')
    
subplot(212)
plot(np.abs(unfiltered_output))
axvline(x=pad,linewidth=2, color='r')
for x in range(N, no_samples-pad,N):
    axvline(x=pad + x,linewidth=1, color='g')
    
    
figure(2)
subplot(211)
title("Hamming window")
plot(w.reshape(P*N))
subplot(212)
title("inverse window")
plot(w_i.reshape(P*N))
  
figure(3)
subplot(211)
title("Filtered Short time fast fourier transforms")
plot(pfb_filtered_output)
axvline(x=pad,linewidth=2, color='r')
for x in range(N, no_samples-pad,N):
    axvline(x=pad + x,linewidth=1, color='g')
subplot(212)
plot(np.abs(pfb_output))
axvline(x=pad,linewidth=2, color='r')
for x in range(N, no_samples-pad,N):
    axvline(x=pad + x,linewidth=1, color='g')

figure(4)
subplot(311)
title("Inverse pfb input")
plot(np.abs(pfb_inverse_input))
axvline(x=pad,linewidth=2, color='r')
for x in range(N, len_of_valid_pfb_output-pad,N):
    axvline(x=pad + x,linewidth=1, color='g')
subplot(312)
title("Short time IFFTs of inverse pfb input")
plot(pfb_inverse_ifft_output)
axvline(x=pad,linewidth=2, color='r')
for x in range(N, len_of_valid_pfb_output-pad,N):
    axvline(x=pad + x,linewidth=1, color='g')
subplot(313)
title("Inverse pfb output")
plot(pfb_inverse_output)
axvline(x=pad,linewidth=2, color='r')
for x in range(N, len_of_valid_pfb_output-pad,N):
    axvline(x=pad + x,linewidth=1, color='g')


num_fourier_samples = len_of_valid_pfb_output-pad
fft_pfb_inv_output = np.fft.fft(pfb_inverse_output[pad:])[0:num_fourier_samples/2]

figure(5)
subplot(211)
title("FFT.r of inverse pfb")
plot(np.arange(0,sampling_freq,sampling_freq/num_fourier_samples)[0:num_fourier_samples/2]/1.0e6,np.abs(np.real(fft_pfb_inv_output))[0:num_fourier_samples/2])
subplot(212)
title("FFT.i of inverse pfb")
plot(np.arange(0,sampling_freq,sampling_freq/num_fourier_samples)[0:num_fourier_samples/2]/1.0e6,np.abs(np.imag(fft_pfb_inv_output))[0:num_fourier_samples/2])
xlabel("F (MHz)")

figure(6)
title("X correlation between tone and pfb^(-1)")
plot(xCor)
xlabel("Sample number")
show()

