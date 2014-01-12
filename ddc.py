'''
Created on Jan 10, 2014

@author: benjamin
'''

import numpy as np
import scipy.signal
from pylab import *
from struct import pack
'''
   ____                       _               
  / __ \                     (_)              
 | |  | |_   _____ _ ____   ___  _____      __
 | |  | \ \ / / _ \ '__\ \ / / |/ _ \ \ /\ / /
 | |__| |\ V /  __/ |   \ V /| |  __/\ V  V / 
  \____/  \_/ \___|_|    \_/ |_|\___| \_/\_/  
                                              
The digital downconvertion (DDC) process is divided into 3 primary stages
1. Mixing
2. Filtering
3. Decimation

Given a band of frequencies ranging from baseband (0 MHz) to some upper limit f_upper, DDC attempts
to extract a subband of frequencies to downsample. It should be noted that a resampling process can 
only upsample and downsample by integer multiples of the current sampling frequency.                                            

To get the best savings in sampling rates the subband is shifted down to baseband and a lowpass filter
is applied to cut away the top part of the remaining frequencies before the sample rate is reduced according
to 2 constraints (f_S is the old sample rate):
 1. f_S' = f_S / decimation_factor where decimation_factor IS AN **INTEGER**
 2. f_S' >= 2 * subband_upper, to comply with the nyquest sampling theorem
 
In order to shift the subband down to baseband it is mixed with a tone of lower frequency, the Local Oscillator
(LO) tone. During mixing the current sampled tone (ST) is element-wise multiplied with the LO-tone where 
||ST|| == ||LO||, such that OutputTone[i] = ST[i] * LO[i]. This mixing has the effect of shifting all frequencies 
within ST to f_ST - f_LO and f_ST + f_LO. The mixing process is hence not a linear process (as would be the case if
two signals are simply added or scalled).

If f_LO is chosen to be the lower limit of the subband we can in essence shift that subband down by mixing the ST and LO
signals. The duplicate band that has been shifted up by f_LO is simply filtered away by applying a lowpass
FIR filter.

We now simply step through the filtered signal keeping every 'decimination_factor'th sample. The new sample rate
is therefore f_S / decimination_factor. Keep in mind this still has to satisfy the nyquest sampling theorem.   

Optimization:
Combine the filtering and decimation steps to only output the convolution at every 'decimination_factor'th sample.
This is possible because in convolution each output value does not depend on any other output value.  
'''

'''
   _____ _       _           _                       _              _       
  / ____| |     | |         | |                     | |            | |      
 | |  __| | ___ | |__   __ _| |   ___ ___  _ __  ___| |_ __ _ _ __ | |_ ___ 
 | | |_ | |/ _ \| '_ \ / _` | |  / __/ _ \| '_ \/ __| __/ _` | '_ \| __/ __|
 | |__| | | (_) | |_) | (_| | | | (_| (_) | | | \__ \ || (_| | | | | |_\__ \
  \_____|_|\___/|_.__/ \__,_|_|  \___\___/|_| |_|___/\__\__,_|_| |_|\__|___/
                                                                            
'''
samplingRate = 800e6 #this should be at least double the highest frequency
nRate = 0.5 * samplingRate #nyquest sampling rate
numSamples = 4096
numFourierSamples = 4096
lookupTableSize = 4096 #size of the sin lookup table

LO_freq = 200e6
toneFreq = [210e6,250e6,320e6] #create a tone consisting of multiple frequencies

decimationFactor = 3
#filter design coefficients
numTaps = 128
cutoffFreq = 120e6 #at shifted highest freq

dumpGeneratedTone = False
dumpGeneratedToneFilename = "pytone.dat"
dumpDecimatedOutput = True
dumpDecimatedOutputFilename = "pydecimated.dat"
'''
  _______                  _____                           _
 |__   __|                / ____|                         | | (_)            
    | | ___  _ __   ___  | |  __  ___ _ __   ___ _ __ __ _| |_ _  ___  _ __  
    | |/ _ \| '_ \ / _ \ | | |_ |/ _ \ '_ \ / _ \ '__/ _` | __| |/ _ \| '_ \ 
    | | (_) | | | |  __/ | |__| |  __/ | | |  __/ | | (_| | |_| | (_) | | | |
    |_|\___/|_| |_|\___|  \_____|\___|_| |_|\___|_|  \__,_|\__|_|\___/|_| |_|
                                                                
8-bit tone generation for both the local oscillator (LO) and fake tone.             
'''

'''                                                                                                                            
Construct a sine wave lookup table (sin(n) n in [0...2*pi]) for tone generation.
This table has lookupTableSize number of entries and the sine
wave is scaled to the maximum amplitude that can be stored in a signed 8-bit integer,
[-127 to 127].

Note that:
 1. the values past 2*PI should be looked up as 2*PI mod lookupTableSize (the sine wave simply wraps after lookupTableSize)
 2. DesiredFreq / SamplingFreq * 2*PI is the phase offset (see www.ti.com/litv/pdf/spra819 for short explanation),
    so using sinTable[n * ( step = floor(tableSize * desiredFreq / sampleFreq)] for n in 0...numSamples, will give an 
    approximation to sin(2*pi*desiredFreq/sampleFreq*n) for n in 0...numSamples
 3. cos(theta) is phase-shifted from sin(theta) by PI/2... to obtain cos from the sin table we shift by
    tableSize/4 
'''
sinTable = (127.0*np.sin(np.arange(0,2*np.pi,2*np.pi/lookupTableSize))).astype(np.float32)
'''
Compute the phase step for looking up the correct values in the table
'''
LOInterval = int (lookupTableSize / float (samplingRate) * LO_freq)
toneInterval = [int (lookupTableSize / float (samplingRate) * tF) for tF in toneFreq] 
n = np.arange(numSamples)
'''
Generate fake tone
since tone generation involves adding up multiple waves of amplitude 127, we have to
scale the frequencies so that each have an equal contribution to the [-127,127] range
'''
scaleFactor = 1/float(len(toneInterval)) 
tone = np.zeros(numSamples)
for i in toneInterval:
    tone += scaleFactor * sinTable[(n*i) % lookupTableSize]
tone = tone.astype(np.int8) #cast to 8bit integer samples

if dumpGeneratedTone:
	print "Dumping generated tone"
	s = "".join([pack('b',x) for x in tone])
	f = open(dumpGeneratedToneFilename,"w")
	f.write(s)
	f.close()

'''
generate lo tone
'''
#construct a sin and cos wave at LO frequency:
sin_lo_n = sinTable[(n*LOInterval) % lookupTableSize]
cos_lo_n = sinTable[((lookupTableSize / 4) + n*LOInterval) % lookupTableSize]

'''
  __  __ _______   __
 |  \/  |_   _\ \ / /
 | \  / | | |  \ V / 
 | |\/| | | |   > <  
 | |  | |_| |_ / . \ 
 |_|  |_|_____/_/ \_\
                     
 Element-wise multiply the two signals to mix them with a cos and sin wave respectively, each
 with lo as its frequency. We have to rescale one of the signals down into floating point space before multiplication
 otherwise we may overflow the 8bit sample sizes
'''
print "Mixing"
iMix = (cos_lo_n/127.0 * tone).astype(np.int8) 
qMix = (sin_lo_n/127.0 * tone).astype(np.int8) 

'''
  ______ _ _ _              _____            _             
 |  ____(_) | |            |  __ \          (_)            
 | |__   _| | |_ ___ _ __  | |  | | ___  ___ _  __ _ _ __  
 |  __| | | | __/ _ \ '__| | |  | |/ _ \/ __| |/ _` | '_ \ 
 | |    | | | ||  __/ |    | |__| |  __/\__ \ | (_| | | | |
 |_|    |_|_|\__\___|_|    |_____/ \___||___/_|\__, |_| |_|
                                                __/ |      
                                               |___/                                                                                         
Creates a lowpass FIR filter to filter at the shifted highest frequency
'''
print "Generating filter"
firFilter = scipy.signal.firwin(numTaps, cutoffFreq/nRate)

'''
  ______ _ _ _            
 |  ____(_) | |           
 | |__   _| | |_ ___ _ __ 
 |  __| | | | __/ _ \ '__|
 | |    | | | ||  __/ |   
 |_|    |_|_|\__\___|_|   
                          
Through optimized convolution. Constructs the convolution machine which computes
sum_(i = 0)^(numSamples-1)(x[y-i]h[i]) for each output sample index y. However, we
only have to do this for the samples we are not discarding in the decimation process.
Therefore we convolve in steps of the decimation factor.
'''
print "Convolving and decimating"
dec_filter_out = np.zeros(ceil(numSamples/float(decimationFactor)), dtype=np.int8)
for i in range(0,numSamples,decimationFactor):
  outval = 0
  for j,tap in enumerate(firFilter):
    idx = i - j
    outval += tap * iMix[idx] if idx >= 0 else 0 #takes care of the first taps-1 samples edge case 
  dec_filter_out[i/decimationFactor] = outval

if dumpDecimatedOutput:
	print "Dumping decimated mix"
	s = "".join([pack('b',x) for x in dec_filter_out])
	f = open(dumpDecimatedOutputFilename,"w")
	f.write(s)
	f.close()

'''
        _       _       
       | |     | |      
  _ __ | | ___ | |_ ___ 
 | '_ \| |/ _ \| __/ __|
 | |_) | | (_) | |_\__ \
 | .__/|_|\___/ \__|___/
 | |                    
 |_|                    
 
'''
freq = np.arange(0,samplingRate,samplingRate / numFourierSamples) / 1.0e6
figure(1)
subplot(211)
title("Fake tone")
plot(tone)
subplot(212)
title("Scaled FFT of Tone")
'''
note that the second halve of the fourier tranform samples are duplicates of the first halve with amplitudes flipped... 
don't display them
'''
plot(freq[:0.5*numFourierSamples], abs(np.imag(np.fft.fft(tone/127.0, numFourierSamples)))[:0.5*numFourierSamples] / numSamples)
xlabel("Frequency (MHz)")

figure(2)
subplot(211)
title("Scaled FFT of iMix (cos wave at lo = "+str(LO_freq/1e6)+" MHz)")
plot(freq[:0.5*numFourierSamples], abs(np.imag(np.fft.fft(iMix/127.0, numFourierSamples)))[:0.5*numFourierSamples] / numSamples)
subplot(212)
title("Scaled FFT of qMix (sin wave at lo = "+str(LO_freq/1e6)+" MHz)")
plot(freq[:0.5*numFourierSamples], abs(np.imag(np.fft.fft(qMix/127.0, numFourierSamples)))[:0.5*numFourierSamples] / numSamples)
xlabel("Frequency (MHz)")

figure(3)
subplot(211)
title(str(numTaps)+" tap lowpass filter with cutoff at "+str(cutoffFreq/1e6)+" MHz)")
plot(firFilter)
subplot(212)
title("Scaled FFT of lowpass FIR filter")
plot(freq[:0.5*numFourierSamples], abs(np.imag(np.fft.fft(firFilter, numFourierSamples)))[:0.5*numFourierSamples] / numSamples)
xlabel("Frequency (MHz)")

figure(4)
title("Scaled FFT of filtered and decimated iMix")
plot(freq[:0.5*numFourierSamples] / decimationFactor, abs(np.real(np.fft.fft(dec_filter_out, numFourierSamples)))[:0.5*numFourierSamples] / numSamples / decimationFactor)
xlabel("Frequency (MHz)")
show()

