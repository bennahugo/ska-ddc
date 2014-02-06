import numpy as np
import scipy.signal
from pylab import *
from struct import unpack

if len(sys.argv) != 3:
    print "Please supply command arguements: 'numSamples' 'input filename'"
    sys.exit(1)

samplingRate = 800e6 #this should be at least double the highest frequency
numSamples = int(sys.argv[1])
decimationFactor = 25
interpFactor = 4
numFourierSamples = 4096
    
#unpack from byte stream:
f = open(sys.argv[2],"r")
s = f.read()
f.close()
dec_filter_out = np.zeros(len(s))
for i,c in enumerate(s):
	dec_filter_out[i] = unpack('b',c)[0]
dec_filter_out = dec_filter_out.astype(np.int8)

freq = np.arange(0,samplingRate,samplingRate / numFourierSamples) / 1.0e6 * interpFactor / decimationFactor

print "x dim", len(freq[:0.5*numFourierSamples])
print "y dim", len(abs(np.real(np.fft.fft(dec_filter_out, numFourierSamples)))[:0.5*numFourierSamples])
print "dec_fil_len", len(dec_filter_out)
figure(1)
title("Scaled FFT of filtered and decimated iMix")
plot(freq[:0.5*numFourierSamples], abs(np.real(np.fft.fft(dec_filter_out, numFourierSamples)))[:0.5*numFourierSamples] / numSamples)
xlabel("Frequency (MHz)")
show() 

