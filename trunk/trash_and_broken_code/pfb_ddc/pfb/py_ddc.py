#!/usr/bin/python
import numpy as np
import scipy.signal
from pylab import *

sample_rate = 800e6
no_samples = 1024
tone_freq = 254e6
 # the tone of interest
lo_freq = 199e6
 # the lo freq used in mixing
decimation = 6
 # decimate sampling. 800e6 / 6 == 133Mhz > 128Mhz

taps = 128
cutoff = 64e6
nyq = sample_rate/2.0
 # filter params

sin_table_length = 4096
 # length of lookup table
no_freqs = 8192
 # number of FFT samples
 
#### end of params

sin_angles = np.arange(0, 2*np.pi, 2*np.pi/sin_table_length)
sin_table = (127 * np.sin(sin_angles)).astype(np.float32)
 # lookup table of sin values scaled to input range

lo_interval = int((sin_table_length / sample_rate) * lo_freq)
tone_interval = int((sin_table_length / sample_rate) * tone_freq)
 # the stepping intervals through the sin table to generate the required frequency
 
print "Adjusted LO freq to", lo_interval * sample_rate / sin_table_length / 1e6, 'MHz'
print "Adjusted tone freq to", tone_interval * sample_rate / sin_table_length / 1e6, 'MHz'
tone_channel = tone_interval * no_freqs / float(sin_table_length)
lo_channel = lo_interval * no_freqs / float(sin_table_length)
tone_channel_after_mix = tone_channel - lo_channel
tone_channel_after_deci = tone_channel_after_mix * decimation
if tone_channel_after_mix < 0:
    tone_channel_after_mix += no_freqs
    tone_channel_after_deci += no_freqs
print "Tone should be in channel %g before mixing, channel %g after mixing and channel %g after decimation" % \
      (tone_channel, tone_channel_after_mix, tone_channel_after_deci)

n = np.arange(no_samples)
 # discrete time indices
tone_data = sin_table[(n*tone_interval) % sin_table_length].astype(np.int8)
 # our 8 bit input data stream that includes the tone...
x = tone_data

b = scipy.signal.firwin(taps, cutoff/nyq)
a = np.zeros(taps)
a[0] = 1
 # low pass filter

sin_lo_n = sin_table[(n*lo_interval) % sin_table_length]
cos_lo_n = sin_table[((sin_table_length / 4) + n*lo_interval) % sin_table_length]
i_mix = (x * (cos_lo_n/127)).astype(np.int8)
q_mix = x * sin_lo_n
i_fir = scipy.signal.lfilter(b, a, i_mix)
q_fir = scipy.signal.lfilter(b, a, q_mix)
 # low pass filtering

i_out = (i_fir).astype(np.int8)[::decimation]
q_out = (q_fir/127).astype(np.int8)[::decimation]
 # finished...

 # try decimating fir filter
i_firdec_out = np.zeros(int(no_samples/decimation)+1, dtype=np.int8)
count = 0
for i in range(0, no_samples, decimation):
    outval = 0
    for j, tap in enumerate(b):
        idx = i - j
        outval += tap * i_mix[idx] if idx >= 0 else 0
    i_firdec_out[count] = outval / 127
    count += 1

# Plot it up :)
freq = np.arange(0, sample_rate, sample_rate/no_freqs) / 1e6

figure(1)
subplot(211)
title("Tone")
plot(tone_data)
subplot(212)
title("Scaled FFT of Tone")
plot(freq, abs(np.fft.fft(tone_data, no_freqs)) / no_samples)
xlabel('Frequency (MHz)')

figure(2)
subplot(311)
title("Scaled FFT of iMix")
plot(freq, abs(np.fft.fft(i_mix, no_freqs)) / 127. / no_samples)
xticks([])
subplot(312)
title("Frequency response of FIR filter")
plot(freq, abs(np.fft.fft(b, no_freqs)))
xticks([])
ylim(0, 1.2)
subplot(313)
title("Scaled FFT of iFir")
plot(freq, abs(np.fft.fft(i_fir, no_freqs)) / 127. / no_samples)
xlabel('Frequency (MHz)')

figure(3)
# subplot(311)
# title("Frequency response of filter after decimation")
# plot(freq / decimation, abs(np.fft.fft(b[((taps // 2) % decimation)::decimation], no_freqs)) * decimation)
# xticks([])
# ylim(0, 1.1 * ylim()[1])
subplot(211)
title("Scaled FFT of iOut")
plot(freq / decimation, abs(np.fft.fft(i_out, no_freqs)) * decimation / no_samples)
xticks([])
subplot(212)
title("Scaled FFT of iFirDecOut")
plot(freq / decimation, abs(np.fft.fft(i_firdec_out, no_freqs)) * decimation / no_samples)
xlabel('Frequency (MHz)')

show()
