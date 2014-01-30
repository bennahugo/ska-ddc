'''
Created on Jan 30, 2014

@author: benjamin

See IPFB GPU code for fuller explanation on the IPFB process
'''
from pylab import *
import sys
import scipy.signal
if len(sys.argv) != 7:
    print "provide ipfb input filename (as output by pfb_generator), prototype FIR filter, output filename for ipfb output, sample count, N and P"
    sys.exit(1)
in_file = sys.argv[1]
filter_file = sys.argv[2]
out_file = sys.argv[3]
no_samples = int(sys.argv[4])
N = int(sys.argv[5])
P = int(sys.argv[6])

pfb_output = np.fromfile(in_file,dtype=np.complex64)
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

pfb_inverse_ifft_output = np.zeros(no_samples+pad).astype(np.float32)
pfb_inverse_output = np.zeros(no_samples).astype(np.float32)

'''
for computational efficiency invert every FFT from the forward process only once... for xx large data
we'll need a ring buffer to store the previous IFFTs
'''
for lB in range(0,no_samples,N):
    pfb_inverse_ifft_output[lB+pad:lB+pad+N] = np.real(np.fft.ifft(pfb_output[lB:lB+N]))

'''
Inverse filterbank
See discussion in ipfb GPU code
'''
for lB in range(0,no_samples,N):
    
    pfb_inverse_output[lB:lB+N] = np.flipud(pfb_inverse_ifft_output[lB:lB+(P*N)].reshape(P,N)*w_i).sum(axis=0)

pfb_inverse_output.astype(np.float32).tofile(out_file)