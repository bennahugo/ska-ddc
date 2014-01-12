#!/usr/bin/python
import numpy as np
import scipy.signal
from struct import pack
import sys

if len(sys.argv) < 2:
    print "Please supply an output filename."
    sys.exit(1)

sample_rate = 800e6
taps = 512
cutoff = 64e6
nyq = sample_rate/2.0
a = np.zeros(taps)
a[0] = 1
 # filter params
b = scipy.signal.firwin(taps, cutoff/nyq)
s = "".join([pack('f',x) for x in b])
f = open(sys.argv[1],"w")
f.write(s)
f.close()
print "Low pass (",taps,"taps ) with cutoff freq",cutoff,"written to",sys.argv[1]
