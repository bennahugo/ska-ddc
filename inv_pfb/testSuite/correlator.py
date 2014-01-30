'''
Created on Jan 30, 2014

@author: benjamin
'''
from pylab import *
import sys
import scipy.signal
if len(sys.argv) != 5:
        print "please specify [input file 1], [input file 2], [whether input file 1 is 'complex'/'real'] and [whether input file 2 is 'complex'/'real']"
        sys.exit(1)
        
if sys.argv[3] == "real":
    inp1 = np.fromfile(sys.argv[1],dtype=np.float32)
    figure(1)
    title("Input 1")
    plot(inp1)
elif sys.argv[3] == "complex":
    inp1 = np.fromfile(sys.argv[1],dtype=np.complex64)
    figure(1)
    subplot(211)
    title("Input 1.r")
    plot(np.real(inp1))
    subplot(212)
    title("Input 1.i")
    plot(np.imag(inp1))
else:
    print "Invalid datatype specification argument 3 should be either 'complex' or 'real'"

if sys.argv[4] == "real":
    inp2 = np.fromfile(sys.argv[2],dtype=np.float32)
    figure(2)
    title("Input 2")
    plot(inp2)
elif sys.argv[4] == "complex":
    inp2 = np.fromfile(sys.argv[2],dtype=np.complex64)
    figure(2)
    subplot(211)
    title("Input 2.r")
    plot(np.real(inp2))
    subplot(212)
    title("Input 2.i")
    plot(np.imag(inp2))
else:
    print "Invalid datatype specification argument 4 should be either 'complex' or 'real'"    
    
xc = scipy.signal.correlate(inp1,inp2)
figure(3)
title("Cross correlation between Input 1 and Input 2")
plot(xc)
show()