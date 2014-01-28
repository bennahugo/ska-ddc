from pylab import *
import sys
import scipy.signal
if len(sys.argv) != 3:
	print "please specify [tone file] and [output file]"
	sys.exit(1)
tone = np.fromfile(sys.argv[1],dtype=np.float32)
output = np.fromfile(sys.argv[2],dtype=np.float32)
xc = scipy.signal.correlate(tone,output)
figure(1)
subplot(211)
title("Tone")
plot(tone)
subplot(212)
title("Output")
plot(output)

figure(2)
title("Cross correlation between tone and output")
plot(xc)
show()
