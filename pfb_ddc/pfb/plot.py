#!/usr/bin/python
from pylab import *
from struct import unpack
import sys

block_size = 1024*1024
loops = 8
N = 512
fold = (N/2)

f = open(sys.argv[1],"r")
s = f.read(loops * block_size)
gpu_output = unpack(str(loops * block_size)+'b',s)
gpu_complex = np.array(gpu_output, dtype=np.float32).view(np.complex64)
gpu_abs = np.abs(gpu_complex)
plot(gpu_abs[0:fold*10])
vlines([x*fold for x in range(10)],0,max(gpu_abs[0:fold*10]))
print "Array size:",gpu_abs.shape,". Folding on",fold,"results in",int(loops*block_size / (2*fold))*fold,"size"
gpu_abs = gpu_abs[:int(loops*block_size / (2*fold))*fold].reshape((-1, fold))

#plot(gpu_abs[0])
#plot(gpu_abs[1])
#plot(gpu_abs[2])
figure()
plot(10 * np.log10(gpu_abs.mean(axis=0) + 1e-20))
#for x in range(loops):
#plot(gpu_abs[1 * int(block_size / (2*fold))])
 # should be first trace from second block
#plot(gpu_abs[2*int(block_size / (2*fold))])
 # should be first trace from second block
show()
