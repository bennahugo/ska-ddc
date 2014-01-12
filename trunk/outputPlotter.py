import numpy as np
import scipy.signal
from pylab import *
from struct import unpack

samplingRate = 800e6 #this should be at least double the highest frequency
numSamples = 4096
decimationFactor = 3
numFourierSamples = 4096

if len(sys.argv) < 2:
    print "Please supply command arguements: input filename"
    sys.exit(1)
    
#unpack from byte stream:
f = open(sys.argv[1],"r")
s = f.read()
f.close()
dec_filter_out = np.zeros(len(s))
for i,c in enumerate(s):
	dec_filter_out[i] = unpack('b',c)[0]
dec_filter_out = dec_filter_out.astype(np.int8)

freq = np.arange(0,samplingRate,samplingRate / numFourierSamples) / 1.0e6

print "x dim", len(freq[:0.5*numFourierSamples])
print "y dim", len(abs(np.real(np.fft.fft(dec_filter_out, numFourierSamples)))[:0.5*numFourierSamples] / float (len(dec_filter_out)))
print "dec_fil_len",len(dec_filter_out)
figure(1)
title("Scaled FFT of filtered and decimated iMix")
plot(freq[:0.5*numFourierSamples] / decimationFactor, abs(np.real(np.fft.fft(dec_filter_out, numFourierSamples)))[:0.5*numFourierSamples] / numSamples / decimationFactor)
xlabel("Frequency (MHz)")
show() 