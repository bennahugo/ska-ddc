#!/usr/bin/python
from pylab import *
from struct import unpack
import sys
f = open(sys.argv[1],"r")
#s = f.read(1024*1024*4)
#gpu_data = unpack('1048576f',s)
#s = f.read(1024*1024*4)
#gpu_filtered = unpack('1048576f',s)
#print gpu_filtered[0:20]
s = f.read(1024*1024*4)
gpu_output = unpack('1048576f',s)
gpu_complex = np.array(gpu_output, dtype=np.float32).view(np.complex64)
gpu_abs = np.abs(gpu_complex)
working = 10 * np.log10(gpu_abs[:524280].reshape((-1,257)).mean(axis=0))
plot(working)
show()
