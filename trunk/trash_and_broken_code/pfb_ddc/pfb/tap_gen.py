#!/usr/bin/python
import numpy as np
import scipy.signal
import sys
N=800
P=8
h0 = scipy.signal.firwin(P * N, 1. / N)
p = N * np.flipud(h0.reshape(P, N))
p.shape
from struct import pack
s = "".join([pack('f',x) for x in p.flatten()])
f = open(sys.argv[1],"w")
f.write(s)
f.close()
print "Channels: %i, Taps: %i, Written to: %s\n" % (N, P, sys.argv[1])
