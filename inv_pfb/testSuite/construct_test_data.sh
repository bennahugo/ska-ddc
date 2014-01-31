#!/bin/bash
N=512
P=8
num_samples_to_use=`expr $N \* $P \* 105`
output_directory=../data_out
tone_type=noise

if [ -d "$output_directory" ]; then
		rm -r $output_directory		
fi

mkdir $output_directory
echo -e "--------------------------------------\nDumping test data into: $output_directory\nTest data characteristics:\n--------------------------------------\nTone type: $tone_type\nN: $N\nP: $P\nNumber of samples: $num_samples_to_use"
#invoke scripts
python tone_generator.py $output_directory/noise.dat $num_samples_to_use $tone_type $N $P
python filter_generator.py $output_directory/prototype_FIR.dat $N $P
python pfb_generator.py $output_directory/noise.dat $output_directory/prototype_FIR.dat $output_directory/pfb.dat $output_directory/unfiltered_ffts.dat $num_samples_to_use $N $P
python ipfb_generator.py $output_directory/pfb.dat $output_directory/prototype_FIR.dat $output_directory/py_ifftedPFB.dat $output_directory/py_ipfb.dat $num_samples_to_use $N $P
echo -e "Test data characteristics:\n--------------------------------------\nTone type: $tone_type\nN: $N\nP: $P\nNumber of samples: $num_samples_to_use" > $output_directory/test_data_characteristics.txt
