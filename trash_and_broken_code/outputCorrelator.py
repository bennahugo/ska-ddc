import numpy as np
import scipy.signal
from pylab import *
from struct import unpack
from struct import pack

if len(sys.argv) != 4:
    print "Please supply command arguements: 'No. Samples' 'signal1 filename' 'signal2 filename'"
    sys.exit(1)

samplingRate = 800e6 #this should be at least double the highest frequency
numSamples = int(sys.argv[1]) 
decimationFactor = 25
interpFactor = 4
numFourierSamples = 4096

plotOff = False
dumpShiftedTone = True
debugShift = 10
    
#unpack from byte stream:
f = open(sys.argv[2],"r")
s = f.read()
f.close()
signal1 = np.zeros(len(s))
for i,c in enumerate(s):
	signal1[i] = unpack('b',c)[0]

f = open(sys.argv[3],"r")
s = f.read()
f.close()
signal2 = np.zeros(len(s))
for i,c in enumerate(s):
	signal2[i] = unpack('b',c)[0]

#debug sanity check
signal3 = np.zeros(numSamples)
for i in range(debugShift,numSamples):
  signal3[i-debugShift] = signal2[i]
if dumpShiftedTone:
  print "Dumping shifted tone"
  s = "".join([pack('b',x) for x in signal3])
  f = open("shiftedTone.dat","w")
  f.write(s)
  f.close()

correlated = np.zeros(numSamples + 1)
maxV = 0
maxA = 0
for i in range(0,numSamples):
  if i % int(numSamples*0.05) == i / numSamples:
    print "CROSS CORRELATION AT ",i / float(numSamples) * 100.0,"%"
  outval = 0
  for j,tap in enumerate(signal2):
    idx = i + j
    outval += tap * signal1[idx] if idx < numSamples else 0 #takes care of the last taps-1 samples edge case 
  correlated[i] = outval
  if i == 0:
    maxV = correlated[i]
  elif correlated[i] > maxV:
    maxV = correlated[i]
    maxA = i
freq = np.arange(0,samplingRate,samplingRate / numFourierSamples) / 1.0e6 * interpFactor / decimationFactor

print "Signal shift by",maxA,"samples (",maxA/samplingRate,"seconds)"

if plotOff:
  print "Requested not to plot"
  sys.exit(0)

figure(1)
title("Cross correlated signals")
plot(correlated)
xlabel("Samples")
show() 

