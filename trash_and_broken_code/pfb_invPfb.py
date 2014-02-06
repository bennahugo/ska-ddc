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
N = 512 #no. SFFT samples
P = 8  #no. subfilters in the bank
window_length = N * P #length of the hamming window
no_bins = N / 2 #real FFTs mirror half the bins
#tone_freq = [max_freq / float(no_bins) * (100)]
#tone_freq = [max_freq / float(no_bins) * (150.3)]
#tone_freq = [max_freq / float(no_bins) * (100.0),max_freq / float(no_bins) * (150.0)] 
tone_freq = [max_freq / float(no_bins) * (100),max_freq / float(no_bins) * (150.45)] 
impulse_shift = N*P + 0.25*N
tone_generation_mode = "noise" #toggle between using a sinusoidal tone @ tone_freq OR generating gausian noise OR impulse (shifted delta)
should_compute_auto_correlate = True
should_x_correlate = True
no_samples = window_length*3   
pad = N*P #pad the tone

'''
generate a fake tone
'''
tone = np.zeros(no_samples)
if tone_generation_mode == "sine": 
    for f in tone_freq:
        for n in np.arange(0,no_samples):
            tone[n] += np.sin(2 * np.pi * n * (f / float(sampling_freq)))
        print "frequency %f MHz should be in channel %f after filtering" % (f / float(1e6),f / (max_freq / float(no_bins)))
elif tone_generation_mode == "noise":
    tone = np.random.randn(no_samples)
elif tone_generation_mode == "impulse":
    print "impulse at sample %d" % impulse_shift
    tone[impulse_shift] = 1
else:
    print "invalid tone generation option"
    exit(1)

'''
FORWARD PFB
'''
print ">>>Computing pfb"
#setup the windowing function
w = scipy.signal.firwin(P * N, 1. / N).reshape(P, N)

#filter with the filterbank
pfb_input = np.zeros(no_samples + pad).astype(np.complex64)

pfb_input[pad:pad+no_samples] = tone
pfb_filtered_output = np.zeros(no_samples).astype(np.complex64)
pfb_output = np.zeros(no_samples).astype(np.complex64)
for lB in range(0,no_samples,N):
    pfb_filtered_output[lB:lB+N] = (pfb_input[lB:lB+(P*N)].reshape(P,N)*w).sum(axis=0) 
    pfb_output[lB:lB+N] = np.fft.fft(pfb_filtered_output[lB:lB+N])
      
print ">>>Computing the comparison unfiltered SFFT" 
'''
take Short Time FFTs without filtering for comparison
'''
unfiltered_output = np.zeros(no_samples).astype(np.complex64)
for lB in range(0,no_samples,N):
    unfiltered_output[lB:lB+N] = np.fft.fft(pfb_input[lB:lB+N])
    
    
'''
INVERSE PFB
'''
print ">>>Computing inverse pfb"
'''
I have a hunch this is wrong: T Karp and N.J. Fliege's suggestion: corresponding analysis and synthesis subfilters are equal

Setup the inverse windowing function (note it should be the flipped impulse responses of the original subfilters,
according to Daniel Zhou, A REVIEW OF POLYPHASE FILTER BANKS AND THEIR APPLICATION
'''
w_i = np.flipud(w)

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

'''
Inverse filterbank
Note the commutator rotates in the opposite direction to the forward process and we'll have to flip the output
when we save to the array.
'''

for lB in range(0,len_of_valid_pfb_output,N):
    pfb_inverse_output[lB:lB+N] = np.flipud(pfb_inverse_ifft_output[lB:lB+(P*N)].reshape(P,N)*w_i).sum(axis=0) 


if should_x_correlate:
   print ">>>Computing X correlation between input signal and pfb inverse output"
   xCor = scipy.signal.correlate(tone,pfb_inverse_output[pad:len_of_valid_pfb_output])
   am = np.argmax(xCor)
   maxV = np.max(xCor)
   print "Signal shift: %d samples" % (am - (len_of_valid_pfb_output-pad-1))

   octile = (len(xCor)//8)
   data_subset = np.zeros(octile*2)
   data_subset[0:octile] = xCor[2*octile:3*octile]
   data_subset[octile:2*octile] = xCor[5*octile:6*octile]
   mean = np.mean(data_subset)
   stddev = np.std(data_subset)
   print "Mean = %f, sample standard deviation = %f, ratio max:(mean+ssd) = %f" % (mean,stddev,maxV/(mean+stddev))

if should_compute_auto_correlate:
   print ">>>Computing auto correlation of input signal"
   input_auto_correlate = scipy.signal.correlate(tone, tone)
   am = np.argmax(input_auto_correlate)
   maxV = np.max(input_auto_correlate) 
   print "Signal shift: %d samples" % (am - (no_samples-1))

   octile = (len(input_auto_correlate)//8)
   data_subset = np.zeros(octile*2)
   data_subset[0:octile] = input_auto_correlate[2*octile:3*octile]
   data_subset[octile:2*octile] = input_auto_correlate[5*octile:6*octile]
   mean = np.mean(data_subset)
   stddev = np.std(data_subset)
   print "Mean = %f, sample standard deviation = %f, ratio max:(mean+ssd) = %f" % (mean,stddev,maxV/(mean+stddev))
  
print ">>>All done! Plotting..."
'''
Plot
'''
figure(1)
subplot(211)
title("Input signal")
plot(pfb_input)
axvline(x=pad,linewidth=2, color='r')
for x in range(N, no_samples-pad,N):
    axvline(x=pad + x,linewidth=1, color='g')
    
subplot(212)
title("Unfiltered Short time fast fourier transforms")
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
title("PFB Filtered Signal")
plot(pfb_filtered_output)
axvline(x=pad,linewidth=2, color='r')
for x in range(N, no_samples-pad,N):
    axvline(x=pad + x,linewidth=1, color='g')
subplot(212)
title("Filtered Short Time Fast Fourier Transforms")
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

if should_x_correlate:
   figure(6)
   title("X correlation between tone and pfb^(-1)")
   plot(xCor)
   xlabel("Sample number")

if should_compute_auto_correlate:
   figure(7)
   title("Auto correlation of tone")
   plot(input_auto_correlate)
   xlabel("Sample number")
show()



