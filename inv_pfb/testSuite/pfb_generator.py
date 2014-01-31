'''
Created on Jan 30, 2014

@author: benjamin

See inverse pfb GPU code for fuller explanation on the PFB process
'''
from pylab import *
import sys
import scipy.signal
if len(sys.argv) != 8:
    print "provide tone input filename, prototype FIR filter, output filename for filtered output, output filename for unfiltered output, sample count, N and P"
    sys.exit(1)
in_file = sys.argv[1]
filter_file = sys.argv[2]
out_filtered_file = sys.argv[3]
out_unfiltered_file = sys.argv[4]
no_samples = int(sys.argv[5])
N = int(sys.argv[6])
P = int(sys.argv[7])

tone = np.fromfile(in_file,dtype=np.float32)
w = np.fromfile(filter_file,dtype=np.float32).reshape(P,N)
'''      
FORWARD PFB
'''
pad = N*P      
print ">>>Computing forward PFB"
pfb_input = np.zeros(no_samples + pad).astype(np.float32)
    
pfb_input[pad:pad+no_samples] = tone
pfb_filtered_output = np.zeros(no_samples).astype(np.float32)
pfb_output = np.zeros(no_samples).astype(np.complex64)
for lB in range(0,no_samples,N):
    pfb_filtered_output[lB:lB+N] = (pfb_input[lB:lB+(P*N)].reshape(P,N)*w).sum(axis=0)
    #normalize the FFT output according to Parseval's Theorem, otherwise we'll be missing a scaling factor in other IFFT implementations than that provided in numpy
    pfb_output[lB:lB+N] = np.fft.fft(pfb_filtered_output[lB:lB+N]) / N 
       
print ">>>Computing the comparison unfiltered SFFT"
'''
take Short Time FFTs without filtering for comparison
'''
unfiltered_output = np.zeros(no_samples).astype(np.complex64)
for lB in range(0,no_samples,N):
    unfiltered_output[lB:lB+N] = np.fft.fft(pfb_input[lB:lB+N])
'''
dump the output
'''
pfb_output.astype(np.complex64).tofile(out_filtered_file)
unfiltered_output.astype(np.complex64).tofile(out_unfiltered_file)
