#!/usr/bin/python

#this supposedly simulates a fake F engine
#TODO: write code to undo what this fake F engine does

from pylab import *
from struct import unpack
import sys
import scipy.signal

block_size = 4096
 # size of block to operate on each loop
loops = 1
N = 512
P=8
 # of taps

if len(sys.argv) < 2:
    print "Please supply an input raw voltage file."
    sys.exit(1)
 # generate taps
h0 = scipy.signal.firwin(P * N, 1. / N)


p = N * np.flipud(h0.reshape(P, N))


f = open(sys.argv[1],"r")
s = f.read(loops * block_size)
 # read in the data

raw_data = np.array(unpack(str(loops * block_size)+'b',s), dtype=np.float32)

pointer = 0

output_buffer = np.zeros(loops * block_size, dtype=np.int8)
post_filt = np.zeros(block_size, dtype=np.float32)
post_fft = np.zeros(block_size, dtype=np.complex64)
buf = np.zeros((P*N) + block_size, dtype=np.float32)
 # buffer padded with P*N zero block
for x in range(loops):
    if pointer == 0:
        buf[(P*N):] = raw_data[pointer:block_size]
    else:
        buf[:] = raw_data[pointer - (P*N):pointer + block_size]

    for y in range(0,block_size,N):
        post_filt[y:y+N] = (buf[y:y+(P*N)].reshape(P,N) * p).sum(axis=0)
         # apply filter taps to P*N block of data
        post_fft[y:y+N] = np.fft.fft(post_filt[y:y+N])
        temp = post_fft[y:y+N][:N/2].view(dtype=np.float32) / N
         # choose first N/2 complex number and normalise
        output_buffer[pointer+y:pointer+y+N] = temp.astype(np.int8)
    pointer += block_size

output_float = output_buffer.astype(np.float32)
output_abs = np.abs(output_float.view(np.complex64))
plot(output_abs[0:N/2*10])
vlines([x*N/2 for x in range(10)],0,max(output_abs[0:N/2*10]))
output_folded = output_abs[:int(loops*block_size / (N))*N/2].reshape((-1, N/2))
figure()
plot(10 * np.log10(output_folded.mean(axis=0) + 1e-20))
show()
