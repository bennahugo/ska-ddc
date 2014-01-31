'''
Created on Jan 30, 2014

@author: benjamin

See IPFB GPU code for fuller explanation on the IPFB process
'''
from pylab import *
import sys
import scipy.signal
if len(sys.argv) != 8:
    print "provide ipfb input filename (as output by pfb_generator), prototype FIR filter, output filename for IFFTed data, output filename for ipfb output, sample count, N and P"
    sys.exit(1)
in_file = sys.argv[1]
filter_file = sys.argv[2]
ifft_out_file = sys.argv[3]
out_file = sys.argv[4]
no_samples = int(sys.argv[5])
N = int(sys.argv[6])
P = int(sys.argv[7])
fft_non_redundant = N / 2 + 1 #N/2+1 of non-redundant samples in a real fft by the Hermite-symmetric property
assert(no_samples % fft_non_redundant == 0) #ensure we're only inverting an integral number of FFTs (with only non-redundant samples)
ipfb_output_size = (no_samples/fft_non_redundant)*N

pfb_output = np.fromfile(in_file,dtype=np.int8).astype(np.float32).view(np.complex64)
w = np.fromfile(filter_file,dtype=np.float32).reshape(P,N)

'''
INVERSE PFB
'''
pad = N*P
print ">>>Computing inverse pfb"
'''
Setup the inverse windowing function (note it should be the flipped impulse responses of the original subfilters,
according to Daniel Zhou, A REVIEW OF POLYPHASE FILTER BANKS AND THEIR APPLICATION
'''
w_i = np.fliplr(w)

pfb_inverse_ifft_output = np.zeros(ipfb_output_size+pad).astype(np.float32)
pfb_inverse_output = np.zeros(ipfb_output_size).astype(np.float32)

'''
for computational efficiency invert every FFT from the forward process only once... for xx large data
we'll need a persistant buffer / ring buffer to store the previous IFFTs -- the buffering approach is explained and implemented in the CUDA version
'''
for lB in range(0,no_samples,fft_non_redundant):
    #reverse to what we've done in the forward pfb: we jump in steps of N on the LHS and steps of N/2 + 1 on the RHS
    output_lB = (lB/fft_non_redundant)*N + pad
    output_uB = output_lB + N
    #reverse the scaling factor (Parseval's Theorem) that we had to add in the forward python PFB to make it compatible with the CUDA real IFFT implementation
    pfb_inverse_ifft_output[output_lB:output_uB] = np.real(np.fft.irfftn(pfb_output[lB:lB+fft_non_redundant] * N))

'''
Inverse filterbank
See discussion in ipfb GPU code
'''
for lB in range(0,ipfb_output_size,N): 
    pfb_inverse_output[lB:lB+N] = np.flipud(pfb_inverse_ifft_output[lB:lB+(P*N)].reshape(P,N)*w_i).sum(axis=0)

pfb_inverse_output.astype(np.float32).astype(np.int8).tofile(out_file)
pfb_inverse_ifft_output.astype(np.float32).astype(np.int8).tofile(ifft_out_file)
